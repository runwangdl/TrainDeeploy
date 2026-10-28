
#ifndef __DEEPLOY_OPTIMIZER_HEADER__
#define __DEEPLOY_OPTIMIZER_HEADER__
#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void RunOptimizerNetwork(uint32_t core_id, uint32_t numThreads);
void InitOptimizerNetwork(uint32_t core_id, uint32_t numThreads);

extern int8_t *DeeployOptNetwork_MEMORYARENA_L1;
static const uint32_t DeeployOptNetwork_MEMORYARENA_L1_len = 14464;
extern int8_t *DeeployOptNetwork_MEMORYARENA_L2;
static const uint32_t DeeployOptNetwork_MEMORYARENA_L2_len = 34632;
extern int8_t *DeeployOptNetwork_input_0;
static const uint32_t DeeployOptNetwork_input_0_len = 32;
extern int32_t *DeeployOptNetwork_input_1;
static const uint32_t DeeployOptNetwork_input_1_len = 8;
extern int8_t *DeeployOptNetwork_input_2;
static const uint32_t DeeployOptNetwork_input_2_len = 2048;
extern int32_t *DeeployOptNetwork_input_3;
static const uint32_t DeeployOptNetwork_input_3_len = 16;
extern int8_t *DeeployOptNetwork_input_4;
static const uint32_t DeeployOptNetwork_input_4_len = 2048;
extern int32_t *DeeployOptNetwork_input_5;
static const uint32_t DeeployOptNetwork_input_5_len = 16;
extern int8_t *DeeployOptNetwork_input_6;
static const uint32_t DeeployOptNetwork_input_6_len = 3584;
extern int32_t *DeeployOptNetwork_input_7;
static const uint32_t DeeployOptNetwork_input_7_len = 32;
extern int8_t *DeeployOptNetwork_input_8;
static const uint32_t DeeployOptNetwork_input_8_len = 7168;
extern int32_t *DeeployOptNetwork_input_9;
static const uint32_t DeeployOptNetwork_input_9_len = 32;
extern float32_t *DeeployOptNetwork_input_10;
static const uint32_t DeeployOptNetwork_input_10_len = 288;
extern float32_t *DeeployOptNetwork_input_11;
static const uint32_t DeeployOptNetwork_input_11_len = 9;
extern float32_t *DeeployOptNetwork_input_12;
static const uint32_t DeeployOptNetwork_input_12_len = 8;
extern float32_t *DeeployOptNetwork_input_13;
static const uint32_t DeeployOptNetwork_input_13_len = 8;
extern float32_t *DeeployOptNetwork_input_14;
static const uint32_t DeeployOptNetwork_input_14_len = 16;
extern float32_t *DeeployOptNetwork_input_15;
static const uint32_t DeeployOptNetwork_input_15_len = 16;
extern float32_t *DeeployOptNetwork_input_16;
static const uint32_t DeeployOptNetwork_input_16_len = 16;
extern float32_t *DeeployOptNetwork_input_17;
static const uint32_t DeeployOptNetwork_input_17_len = 16;
extern float32_t *DeeployOptNetwork_input_18;
static const uint32_t DeeployOptNetwork_input_18_len = 32;
extern float32_t *DeeployOptNetwork_input_19;
static const uint32_t DeeployOptNetwork_input_19_len = 32;
extern float32_t *DeeployOptNetwork_input_20;
static const uint32_t DeeployOptNetwork_input_20_len = 32;
extern float32_t *DeeployOptNetwork_input_21;
static const uint32_t DeeployOptNetwork_input_21_len = 32;
extern int8_t *DeeployOptNetwork_output_0;
static const uint32_t DeeployOptNetwork_output_0_len = 32;
extern int32_t *DeeployOptNetwork_output_1;
static const uint32_t DeeployOptNetwork_output_1_len = 8;
extern int8_t *DeeployOptNetwork_output_2;
static const uint32_t DeeployOptNetwork_output_2_len = 2048;
extern int32_t *DeeployOptNetwork_output_3;
static const uint32_t DeeployOptNetwork_output_3_len = 16;
extern int8_t *DeeployOptNetwork_output_4;
static const uint32_t DeeployOptNetwork_output_4_len = 2048;
extern int32_t *DeeployOptNetwork_output_5;
static const uint32_t DeeployOptNetwork_output_5_len = 16;
extern int8_t *DeeployOptNetwork_output_6;
static const uint32_t DeeployOptNetwork_output_6_len = 3584;
extern int32_t *DeeployOptNetwork_output_7;
static const uint32_t DeeployOptNetwork_output_7_len = 32;
extern int8_t *DeeployOptNetwork_output_8;
static const uint32_t DeeployOptNetwork_output_8_len = 7168;
extern int32_t *DeeployOptNetwork_output_9;
static const uint32_t DeeployOptNetwork_output_9_len = 32;
extern float32_t *DeeployOptNetwork_output_10;
static const uint32_t DeeployOptNetwork_output_10_len = 288;
extern float32_t *DeeployOptNetwork_output_11;
static const uint32_t DeeployOptNetwork_output_11_len = 9;
extern float32_t *DeeployOptNetwork_output_12;
static const uint32_t DeeployOptNetwork_output_12_len = 8;
extern float32_t *DeeployOptNetwork_output_13;
static const uint32_t DeeployOptNetwork_output_13_len = 8;
extern float32_t *DeeployOptNetwork_output_14;
static const uint32_t DeeployOptNetwork_output_14_len = 16;
extern float32_t *DeeployOptNetwork_output_15;
static const uint32_t DeeployOptNetwork_output_15_len = 16;
extern float32_t *DeeployOptNetwork_output_16;
static const uint32_t DeeployOptNetwork_output_16_len = 16;
extern float32_t *DeeployOptNetwork_output_17;
static const uint32_t DeeployOptNetwork_output_17_len = 16;
extern float32_t *DeeployOptNetwork_output_18;
static const uint32_t DeeployOptNetwork_output_18_len = 32;
extern float32_t *DeeployOptNetwork_output_19;
static const uint32_t DeeployOptNetwork_output_19_len = 32;
extern float32_t *DeeployOptNetwork_output_20;
static const uint32_t DeeployOptNetwork_output_20_len = 32;
extern float32_t *DeeployOptNetwork_output_21;
static const uint32_t DeeployOptNetwork_output_21_len = 32;
static const uint32_t DeeployOptNetwork_num_inputs = 22;
static const uint32_t DeeployOptNetwork_num_outputs = 22;
extern void *DeeployOptNetwork_inputs[22];
extern void *DeeployOptNetwork_outputs[22];
static const uint32_t DeeployOptNetwork_inputs_bytes[22] = {32, 32, 2048, 64, 2048, 64, 3584, 128, 7168, 128, 1152,
                                                            36, 32, 32,   64, 64,   64, 64,   128, 128,  128, 128};
static const uint32_t DeeployOptNetwork_outputs_bytes[22] = {32, 32, 2048, 64, 2048, 64, 3584, 128, 7168, 128, 1152,
                                                             36, 32, 32,   64, 64,   64, 64,   128, 128,  128, 128};
#endif
