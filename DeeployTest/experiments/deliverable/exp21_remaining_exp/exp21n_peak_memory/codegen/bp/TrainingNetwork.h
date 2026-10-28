
#ifndef __DEEPLOY_TRAINING_HEADER__
#define __DEEPLOY_TRAINING_HEADER__
#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "perf_utils.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void RunTrainingNetwork(uint32_t core_id, uint32_t numThreads);
void InitTrainingNetwork(uint32_t core_id, uint32_t numThread);

extern int8_t *DeeployNetwork_MEMORYARENA_L1;
static const uint32_t DeeployNetwork_MEMORYARENA_L1_len = 121984;
extern int8_t *DeeployNetwork_MEMORYARENA_L2;
static const uint32_t DeeployNetwork_MEMORYARENA_L2_len = 1478772;
extern float32_t *DeeployNetwork_input_0;
static const uint32_t DeeployNetwork_input_0_len = 9800;
extern int64_t *DeeployNetwork_input_1;
static const uint32_t DeeployNetwork_input_1_len = 1;
extern float32_t *DeeployNetwork_input_2;
static const uint32_t DeeployNetwork_input_2_len = 32;
extern float32_t *DeeployNetwork_input_3;
static const uint32_t DeeployNetwork_input_3_len = 8;
extern float32_t *DeeployNetwork_input_4;
static const uint32_t DeeployNetwork_input_4_len = 8;
extern float32_t *DeeployNetwork_input_5;
static const uint32_t DeeployNetwork_input_5_len = 8;
extern float32_t *DeeployNetwork_input_6;
static const uint32_t DeeployNetwork_input_6_len = 2048;
extern float32_t *DeeployNetwork_input_7;
static const uint32_t DeeployNetwork_input_7_len = 16;
extern float32_t *DeeployNetwork_input_8;
static const uint32_t DeeployNetwork_input_8_len = 16;
extern float32_t *DeeployNetwork_input_9;
static const uint32_t DeeployNetwork_input_9_len = 16;
extern float32_t *DeeployNetwork_input_10;
static const uint32_t DeeployNetwork_input_10_len = 2048;
extern float32_t *DeeployNetwork_input_11;
static const uint32_t DeeployNetwork_input_11_len = 16;
extern float32_t *DeeployNetwork_input_12;
static const uint32_t DeeployNetwork_input_12_len = 16;
extern float32_t *DeeployNetwork_input_13;
static const uint32_t DeeployNetwork_input_13_len = 16;
extern float32_t *DeeployNetwork_input_14;
static const uint32_t DeeployNetwork_input_14_len = 3584;
extern float32_t *DeeployNetwork_input_15;
static const uint32_t DeeployNetwork_input_15_len = 32;
extern float32_t *DeeployNetwork_input_16;
static const uint32_t DeeployNetwork_input_16_len = 32;
extern float32_t *DeeployNetwork_input_17;
static const uint32_t DeeployNetwork_input_17_len = 32;
extern float32_t *DeeployNetwork_input_18;
static const uint32_t DeeployNetwork_input_18_len = 7168;
extern float32_t *DeeployNetwork_input_19;
static const uint32_t DeeployNetwork_input_19_len = 32;
extern float32_t *DeeployNetwork_input_20;
static const uint32_t DeeployNetwork_input_20_len = 32;
extern float32_t *DeeployNetwork_input_21;
static const uint32_t DeeployNetwork_input_21_len = 32;
extern float32_t *DeeployNetwork_input_22;
static const uint32_t DeeployNetwork_input_22_len = 288;
extern float32_t *DeeployNetwork_input_23;
static const uint32_t DeeployNetwork_input_23_len = 9;
extern float32_t *DeeployNetwork_input_24;
static const uint32_t DeeployNetwork_input_24_len = 32;
extern float32_t *DeeployNetwork_input_25;
static const uint32_t DeeployNetwork_input_25_len = 8;
extern float32_t *DeeployNetwork_input_26;
static const uint32_t DeeployNetwork_input_26_len = 8;
extern float32_t *DeeployNetwork_input_27;
static const uint32_t DeeployNetwork_input_27_len = 8;
extern float32_t *DeeployNetwork_input_28;
static const uint32_t DeeployNetwork_input_28_len = 2048;
extern float32_t *DeeployNetwork_input_29;
static const uint32_t DeeployNetwork_input_29_len = 16;
extern float32_t *DeeployNetwork_input_30;
static const uint32_t DeeployNetwork_input_30_len = 16;
extern float32_t *DeeployNetwork_input_31;
static const uint32_t DeeployNetwork_input_31_len = 16;
extern float32_t *DeeployNetwork_input_32;
static const uint32_t DeeployNetwork_input_32_len = 2048;
extern float32_t *DeeployNetwork_input_33;
static const uint32_t DeeployNetwork_input_33_len = 16;
extern float32_t *DeeployNetwork_input_34;
static const uint32_t DeeployNetwork_input_34_len = 16;
extern float32_t *DeeployNetwork_input_35;
static const uint32_t DeeployNetwork_input_35_len = 16;
extern float32_t *DeeployNetwork_input_36;
static const uint32_t DeeployNetwork_input_36_len = 3584;
extern float32_t *DeeployNetwork_input_37;
static const uint32_t DeeployNetwork_input_37_len = 32;
extern float32_t *DeeployNetwork_input_38;
static const uint32_t DeeployNetwork_input_38_len = 32;
extern float32_t *DeeployNetwork_input_39;
static const uint32_t DeeployNetwork_input_39_len = 32;
extern float32_t *DeeployNetwork_input_40;
static const uint32_t DeeployNetwork_input_40_len = 7168;
extern float32_t *DeeployNetwork_input_41;
static const uint32_t DeeployNetwork_input_41_len = 32;
extern float32_t *DeeployNetwork_input_42;
static const uint32_t DeeployNetwork_input_42_len = 32;
extern float32_t *DeeployNetwork_input_43;
static const uint32_t DeeployNetwork_input_43_len = 32;
extern float32_t *DeeployNetwork_input_44;
static const uint32_t DeeployNetwork_input_44_len = 288;
extern float32_t *DeeployNetwork_input_45;
static const uint32_t DeeployNetwork_input_45_len = 9;
extern uint8_t *DeeployNetwork_input_46;
static const uint32_t DeeployNetwork_input_46_len = 1;
extern float32_t *DeeployNetwork_output_0;
static const uint32_t DeeployNetwork_output_0_len = 1.0;
extern float32_t *DeeployNetwork_output_1;
static const uint32_t DeeployNetwork_output_1_len = 32;
extern float32_t *DeeployNetwork_output_2;
static const uint32_t DeeployNetwork_output_2_len = 8;
extern float32_t *DeeployNetwork_output_3;
static const uint32_t DeeployNetwork_output_3_len = 8;
extern float32_t *DeeployNetwork_output_4;
static const uint32_t DeeployNetwork_output_4_len = 8;
extern float32_t *DeeployNetwork_output_5;
static const uint32_t DeeployNetwork_output_5_len = 2048;
extern float32_t *DeeployNetwork_output_6;
static const uint32_t DeeployNetwork_output_6_len = 16;
extern float32_t *DeeployNetwork_output_7;
static const uint32_t DeeployNetwork_output_7_len = 16;
extern float32_t *DeeployNetwork_output_8;
static const uint32_t DeeployNetwork_output_8_len = 16;
extern float32_t *DeeployNetwork_output_9;
static const uint32_t DeeployNetwork_output_9_len = 2048;
extern float32_t *DeeployNetwork_output_10;
static const uint32_t DeeployNetwork_output_10_len = 16;
extern float32_t *DeeployNetwork_output_11;
static const uint32_t DeeployNetwork_output_11_len = 16;
extern float32_t *DeeployNetwork_output_12;
static const uint32_t DeeployNetwork_output_12_len = 16;
extern float32_t *DeeployNetwork_output_13;
static const uint32_t DeeployNetwork_output_13_len = 3584;
extern float32_t *DeeployNetwork_output_14;
static const uint32_t DeeployNetwork_output_14_len = 32;
extern float32_t *DeeployNetwork_output_15;
static const uint32_t DeeployNetwork_output_15_len = 32;
extern float32_t *DeeployNetwork_output_16;
static const uint32_t DeeployNetwork_output_16_len = 32;
extern float32_t *DeeployNetwork_output_17;
static const uint32_t DeeployNetwork_output_17_len = 7168;
extern float32_t *DeeployNetwork_output_18;
static const uint32_t DeeployNetwork_output_18_len = 32;
extern float32_t *DeeployNetwork_output_19;
static const uint32_t DeeployNetwork_output_19_len = 32;
extern float32_t *DeeployNetwork_output_20;
static const uint32_t DeeployNetwork_output_20_len = 32;
extern float32_t *DeeployNetwork_output_21;
static const uint32_t DeeployNetwork_output_21_len = 288;
extern float32_t *DeeployNetwork_output_22;
static const uint32_t DeeployNetwork_output_22_len = 9;
static const uint32_t DeeployNetwork_num_inputs = 47;
static const uint32_t DeeployNetwork_num_outputs = 23;
extern void *DeeployNetwork_inputs[47];
extern void *DeeployNetwork_outputs[23];
static const uint32_t DeeployNetwork_inputs_bytes[47] = {39200, 8,   128,   32,  32,    32,  8192, 64,  64,    64,  8192, 64,  64,   64, 14336, 128,
                                                         128,   128, 28672, 128, 128,   128, 1152, 36,  128,   32,  32,   32,  8192, 64, 64,    64,
                                                         8192,  64,  64,    64,  14336, 128, 128,  128, 28672, 128, 128,  128, 1152, 36, 1};
static const uint32_t DeeployNetwork_inputs_elems[47] = {9800, 1,  32,   8,  8,    8,  2048, 16, 16,   16, 2048, 16, 16,   16, 3584, 32,
                                                         32,   32, 7168, 32, 32,   32, 288,  9,  32,   8,  8,    8,  2048, 16, 16,   16,
                                                         2048, 16, 16,   16, 3584, 32, 32,   32, 7168, 32, 32,   32, 288,  9,  1};
static const uint32_t DeeployNetwork_outputs_bytes[23] = {4.0, 128,   32,  32,  32,  8192,  64,  64,  64,  8192, 64, 64,
                                                          64,  14336, 128, 128, 128, 28672, 128, 128, 128, 1152, 36};
#endif
