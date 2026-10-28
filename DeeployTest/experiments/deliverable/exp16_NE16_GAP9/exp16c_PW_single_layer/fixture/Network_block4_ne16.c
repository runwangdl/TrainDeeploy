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

static PI_L2 int32_t DeeployNetwork_w_pmul_tensor[32] = {241188, 288669, 244036, 253502, 286550, 323039, 233256, 257786, 249270, 230927, 310926,
                                                         246525, 244214, 212394, 233318, 332178, 260461, 188270, 258577, 273159, 206862, 260937,
                                                         193834, 192672, 235062, 274247, 226927, 202666, 266184, 247421, 208752, 158957};

static PI_L2 int32_t DeeployNetwork_rqs_mul_tensor[32] = {122, 102, 120, 116, 102, 91,  126, 114, 118, 127, 94,  119, 120, 138, 126, 88,
                                                          113, 156, 113, 107, 142, 112, 151, 152, 125, 107, 129, 145, 110, 119, 140, 184};

static PI_L2 int32_t DeeployNetwork_rqs_add_tensor[32] = {32640, 32615, 32898, 32633, 32616, 32940, 32892, 32905, 32900, 32891, 32933,
                                                          32899, 32638, 32655, 32644, 32591, 32906, 32668, 32631, 32913, 32658, 32907,
                                                          32665, 32666, 32893, 32914, 32647, 32876, 32627, 32637, 32879, 32684};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_cumByteOffset[1] = {0};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride[2] = {96, 64};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride[2] = {384, 256};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo[2] = {3, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi[2] = {3, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_numTiles[2] = {0, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_cumByteOffset[2] = {0, 96};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_cmd[2] = {1966848, 1966592};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_size[2] = {768, 512};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_size_1d[2] = {96, 64};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_cumByteOffset[2] = {0, 0};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_cumByteOffset[2] = {0, 384};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_cmd[2] = {1835776, 1835520};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_size[2] = {768, 512};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_size_1d[2] = {384, 256};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[2];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref;
} _b4_wpert_tiling_closure_args_t;

static void _b4_wpert_tiling_closure(void *_b4_wpert_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4_wpert_tiling_closure_args_t *args = (_b4_wpert_tiling_closure_args_t *)_b4_wpert_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: b4_wpert, Op: RQSPerturbRademacher)
  uint8_t b4_wpert_core_id = (uint8_t)pi_core_id();
  uint8_t b4_wpert_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t b4_wpert_chunk = (7168 >> b4_wpert_log2Core) + ((7168 & (NUM_CORES - 1)) != 0);
  uint32_t b4_wpert_chunk_start = (uint32_t)MIN(b4_wpert_chunk * b4_wpert_core_id, (uint32_t)7168);
  uint32_t b4_wpert_chunk_stop = (uint32_t)MIN(b4_wpert_chunk_start + b4_wpert_chunk, (uint32_t)7168);
  uint32_t b4_wpert_local_size = b4_wpert_chunk_stop - b4_wpert_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t b4_wpert_channel_start_offset = b4_wpert_chunk_start % 224;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 8 + b4_wpert_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t b4_wpert_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  ApplyPerturbQuantRademacher_CHW((const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref[b4_wpert_chunk_start],
                                  (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref[b4_wpert_chunk_start],
                                  (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref, 15, 224, chunk_seed, b4_wpert_local_size,
                                  b4_wpert_chunk_start,
                                  perturbation_sign,   // -- QW: +eps (L+) / -eps (L-)
                                  b4_wpert_eps_scale); // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref;
} _b4_wpert_cluster_fork_args_t;

static void _b4_wpert_cluster_fork(void *_b4_wpert_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4_wpert_cluster_fork_args_t *args = (_b4_wpert_cluster_fork_args_t *)_b4_wpert_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4_wpert_tiling_closure_args_t DeeployNetwork__b4_wpert_tiling_closure_args =
      (_b4_wpert_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref};

  // _b4_wpert_tiling_closure CLOSURE CALL
  _b4_wpert_tiling_closure(&DeeployNetwork__b4_wpert_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr;
} _b4_wpert_closure_args_t;

static void _b4_wpert_closure(void *_b4_wpert_closure_args) {
  // CLOSURE ARG CAST
  _b4_wpert_closure_args_t *args = (_b4_wpert_closure_args_t *)_b4_wpert_closure_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14336);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_w_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_w_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_weight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref, DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_w_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b4_wpert_cluster_fork_args_t DeeployNetwork__b4_wpert_cluster_fork_args =
        (_b4_wpert_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_in_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_mul_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4_wpert_cluster_fork, &DeeployNetwork__b4_wpert_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1317888, DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_weight_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr;
} _b4_wpert_closure_L3_args_t;

static void _b4_wpert_closure_L3(void *_b4_wpert_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4_wpert_closure_L3_args_t *args = (_b4_wpert_closure_L3_args_t *)_b4_wpert_closure_L3_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4_wpert_closure_args_t DeeployNetwork__b4_wpert_closure_args =
      (_b4_wpert_closure_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                 .DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr};

  // _b4_wpert_closure CLOSURE CALL
  _b4_wpert_closure(&DeeployNetwork__b4_wpert_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref;
} _b4_wenc_tiling_closure_args_t;

static void _b4_wenc_tiling_closure(void *_b4_wenc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4_wenc_tiling_closure_args_t *args = (_b4_wenc_tiling_closure_args_t *)_b4_wenc_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16WeightEncode (Name: b4_wenc, Op: NE16WeightEncode) -- bit-serial encode on the cluster -- QW
  {
    uint32_t b4_wenc_nrows = 224;
    uint32_t b4_wenc_chunk = (b4_wenc_nrows + NUM_CORES - 1) / NUM_CORES;
    uint32_t b4_wenc_start = MIN(b4_wenc_chunk * (uint32_t)pi_core_id(), b4_wenc_nrows);
    uint32_t b4_wenc_stop = MIN(b4_wenc_start + b4_wenc_chunk, b4_wenc_nrows);

    NE16WeightEncode_i8_u8((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref,
                           (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref, 32, 32, 7, b4_wenc_start, b4_wenc_stop - b4_wenc_start);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref;
} _b4_wenc_cluster_fork_args_t;

static void _b4_wenc_cluster_fork(void *_b4_wenc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4_wenc_cluster_fork_args_t *args = (_b4_wenc_cluster_fork_args_t *)_b4_wenc_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4_wenc_tiling_closure_args_t DeeployNetwork__b4_wenc_tiling_closure_args =
      (_b4_wenc_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref};

  // _b4_wenc_tiling_closure CLOSURE CALL
  _b4_wenc_tiling_closure(&DeeployNetwork__b4_wenc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr;
} _b4_wenc_closure_args_t;

static void _b4_wenc_closure(void *_b4_wenc_closure_args) {
  // CLOSURE ARG CAST
  _b4_wenc_closure_args_t *args = (_b4_wenc_closure_args_t *)_b4_wenc_closure_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_weight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_weight_enc_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_enc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_weight_pert_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b4_wenc_cluster_fork_args_t DeeployNetwork__b4_wenc_cluster_fork_args =
        (_b4_wenc_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_in_ref,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4_wenc_cluster_fork, &DeeployNetwork__b4_wenc_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1317888, DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_weight_enc_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr;
} _b4_wenc_closure_L3_args_t;

static void _b4_wenc_closure_L3(void *_b4_wenc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4_wenc_closure_L3_args_t *args = (_b4_wenc_closure_L3_args_t *)_b4_wenc_closure_L3_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4_wenc_closure_args_t DeeployNetwork__b4_wenc_closure_args =
      (_b4_wenc_closure_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                .DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                .DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr};

  // _b4_wenc_closure CLOSURE CALL
  _b4_wenc_closure(&DeeployNetwork__b4_wenc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref;
} _b4_conv1xk_tiling_closure_args_t;

static void _b4_conv1xk_tiling_closure(void *_b4_conv1xk_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4_conv1xk_tiling_closure_args_t *args = (_b4_conv1xk_tiling_closure_args_t *)_b4_conv1xk_tiling_closure_args;

  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16 1x7 dense conv as 7 pointwise dispatches (streamin accumulation)
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref + 0 * 1024,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref + 0 * *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 32, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 32, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 2), .HoWo = nnx_concat_half(1, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref + 1 * 1024,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref + 1 * *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 32, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 32, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 2), .HoWo = nnx_concat_half(1, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref + 2 * 1024,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref + 2 * *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 32, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 32, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 2), .HoWo = nnx_concat_half(1, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref + 3 * 1024,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref + 3 * *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 32, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 32, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 2), .HoWo = nnx_concat_half(1, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref + 4 * 1024,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref + 4 * *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 32, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 32, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 2), .HoWo = nnx_concat_half(1, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref + 5 * 1024,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref + 5 * *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 32, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 32, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 2), .HoWo = nnx_concat_half(1, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref + 6 * 1024,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref + 6 * *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 32, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 32, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 2), .HoWo = nnx_concat_half(1, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref)}},
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
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr;
} _b4_conv1xk_closure_args_t;

static void _b4_conv1xk_closure(void *_b4_conv1xk_closure_args) {
  // CLOSURE ARG CAST
  _b4_conv1xk_closure_args_t *args = (_b4_conv1xk_closure_args_t *)_b4_conv1xk_closure_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride + 0);
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7936);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_enc_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_enc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_input_0_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_size_1d[TILING_I],
                                      .ext_stride_1d = 160};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_input_0_ref
    DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_input_0_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_input_0_ref) + 96);

    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1448960,
                                      .size = 7168,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_enc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref
    DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref = &DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref = &DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref = &DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride[TILING_I];

    _b4_conv1xk_tiling_closure_args_t DeeployNetwork__b4_conv1xk_tiling_closure_args = (_b4_conv1xk_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_in_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_dim_im_out_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_bWi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref};

    // _b4_conv1xk_tiling_closure CLOSURE CALL
    _b4_conv1xk_tiling_closure(&DeeployNetwork__b4_conv1xk_tiling_closure_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_conv_out_tensor_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_data_out_size_1d[TILING_I],
                                      .ext_stride_1d = 640};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_conv_out_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_conv_out_tensor_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_conv_out_tensor_ref) + 384);

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr;
} _b4_conv1xk_closure_L3_args_t;

static void _b4_conv1xk_closure_L3(void *_b4_conv1xk_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4_conv1xk_closure_L3_args_t *args = (_b4_conv1xk_closure_L3_args_t *)_b4_conv1xk_closure_L3_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4_conv1xk_closure_args_t DeeployNetwork__b4_conv1xk_closure_args =
      (_b4_conv1xk_closure_args_t){.DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                   .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                   .DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr};

  // _b4_conv1xk_closure CLOSURE CALL
  _b4_conv1xk_closure(&DeeployNetwork__b4_conv1xk_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref;
} _b4_rqs_tiling_closure_args_t;

static void _b4_rqs_tiling_closure(void *_b4_rqs_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4_rqs_tiling_closure_args_t *args = (_b4_rqs_tiling_closure_args_t *)_b4_rqs_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL

  // RequantShift (Name: b4_rqs, Op: RequantShift)
  RequantShift_s32_s8_NHWC(DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref, 320, DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref,
                           DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref, 16, 32, 0, 0, -128, 127, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref;
} _b4_rqs_cluster_fork_args_t;

static void _b4_rqs_cluster_fork(void *_b4_rqs_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4_rqs_cluster_fork_args_t *args = (_b4_rqs_cluster_fork_args_t *)_b4_rqs_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4_rqs_tiling_closure_args_t DeeployNetwork__b4_rqs_tiling_closure_args =
      (_b4_rqs_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref};

  // _b4_rqs_tiling_closure CLOSURE CALL
  _b4_rqs_tiling_closure(&DeeployNetwork__b4_rqs_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr;
} _b4_rqs_closure_args_t;

static void _b4_rqs_closure(void *_b4_rqs_closure_args) {
  // CLOSURE ARG CAST
  _b4_rqs_closure_args_t *args = (_b4_rqs_closure_args_t *)_b4_rqs_closure_args;

  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 320);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1600);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1728);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_rqs_add_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_add_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_rqs_mul_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_mul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1443072,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_conv_out_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_rqs_add_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_rqs_mul_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _b4_rqs_cluster_fork_args_t DeeployNetwork__b4_rqs_cluster_fork_args =
        (_b4_rqs_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4_rqs_cluster_fork, &DeeployNetwork__b4_rqs_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1311040,
                                      .size = 320,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr;
} _b4_rqs_closure_L3_args_t;

static void _b4_rqs_closure_L3(void *_b4_rqs_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4_rqs_closure_L3_args_t *args = (_b4_rqs_closure_L3_args_t *)_b4_rqs_closure_L3_args;

  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4_rqs_closure_args_t DeeployNetwork__b4_rqs_closure_args =
      (_b4_rqs_closure_args_t){.DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                               .DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr};

  // _b4_rqs_closure CLOSURE CALL
  _b4_rqs_closure(&DeeployNetwork__b4_rqs_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr;
    DeeployNetwork_weight_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 1280);
    _b4_wpert_closure_L3_args_t DeeployNetwork__b4_wpert_closure_L3_args =
        (_b4_wpert_closure_L3_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4_wpert_tileIdxPtr};

    // _b4_wpert_closure_L3 CLOSURE CALL
    _b4_wpert_closure_L3(&DeeployNetwork__b4_wpert_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr;
    DeeployNetwork_weight_enc_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8448);
    _b4_wenc_closure_L3_args_t DeeployNetwork__b4_wenc_closure_L3_args =
        (_b4_wenc_closure_L3_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                     .DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4_wenc_tileIdxPtr};

    // _b4_wenc_closure_L3 CLOSURE CALL
    _b4_wenc_closure_L3(&DeeployNetwork__b4_wenc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr;
    DeeployNetwork_conv_out_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 1280);
    _b4_conv1xk_closure_L3_args_t DeeployNetwork__b4_conv1xk_closure_L3_args =
        (_b4_conv1xk_closure_L3_args_t){.DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                        .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4_conv1xk_tileIdxPtr};

    // _b4_conv1xk_closure_L3 CLOSURE CALL
    _b4_conv1xk_closure_L3(&DeeployNetwork__b4_conv1xk_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr;
    _b4_rqs_closure_L3_args_t DeeployNetwork__b4_rqs_closure_L3_args =
        (_b4_rqs_closure_L3_args_t){.DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                    .DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4_rqs_tileIdxPtr};

    // _b4_rqs_closure_L3 CLOSURE CALL
    _b4_rqs_closure_L3(&DeeployNetwork__b4_rqs_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 14464);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 15616);

  DeeployNetwork_input_0 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8448);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
