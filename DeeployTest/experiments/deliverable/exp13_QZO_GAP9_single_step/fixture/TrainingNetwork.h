
#ifndef __DEEPLOY_TRAINING_HEADER__
#define __DEEPLOY_TRAINING_HEADER__
#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void RunTrainingNetwork(uint32_t core_id, uint32_t numThreads);
void InitTrainingNetwork(uint32_t core_id, uint32_t numThread);

extern int8_t *DeeployNetwork_MEMORYARENA_L1;
static const uint32_t DeeployNetwork_MEMORYARENA_L1_len = 127680;
extern int8_t *DeeployNetwork_MEMORYARENA_L2;
static const uint32_t DeeployNetwork_MEMORYARENA_L2_len = 777980;
extern float32_t *DeeployNetwork_input_0;
static const uint32_t DeeployNetwork_input_0_len = 9800;
extern int64_t *DeeployNetwork_input_1;
static const uint32_t DeeployNetwork_input_1_len = 1;
extern float32_t *DeeployNetwork_input_2;
static const uint32_t DeeployNetwork_input_2_len = 8;
extern float32_t *DeeployNetwork_input_3;
static const uint32_t DeeployNetwork_input_3_len = 8;
extern float32_t *DeeployNetwork_input_4;
static const uint32_t DeeployNetwork_input_4_len = 16;
extern float32_t *DeeployNetwork_input_5;
static const uint32_t DeeployNetwork_input_5_len = 16;
extern float32_t *DeeployNetwork_input_6;
static const uint32_t DeeployNetwork_input_6_len = 16;
extern float32_t *DeeployNetwork_input_7;
static const uint32_t DeeployNetwork_input_7_len = 16;
extern float32_t *DeeployNetwork_input_8;
static const uint32_t DeeployNetwork_input_8_len = 32;
extern float32_t *DeeployNetwork_input_9;
static const uint32_t DeeployNetwork_input_9_len = 32;
extern float32_t *DeeployNetwork_input_10;
static const uint32_t DeeployNetwork_input_10_len = 32;
extern float32_t *DeeployNetwork_input_11;
static const uint32_t DeeployNetwork_input_11_len = 32;
extern int8_t *DeeployNetwork_input_12;
static const uint32_t DeeployNetwork_input_12_len = 32;
extern int8_t *DeeployNetwork_input_13;
static const uint32_t DeeployNetwork_input_13_len = 2048;
extern int8_t *DeeployNetwork_input_14;
static const uint32_t DeeployNetwork_input_14_len = 2048;
extern int8_t *DeeployNetwork_input_15;
static const uint32_t DeeployNetwork_input_15_len = 3584;
extern int8_t *DeeployNetwork_input_16;
static const uint32_t DeeployNetwork_input_16_len = 7168;
extern int32_t *DeeployNetwork_input_17;
static const uint32_t DeeployNetwork_input_17_len = 8;
extern int32_t *DeeployNetwork_input_18;
static const uint32_t DeeployNetwork_input_18_len = 16;
extern int32_t *DeeployNetwork_input_19;
static const uint32_t DeeployNetwork_input_19_len = 16;
extern int32_t *DeeployNetwork_input_20;
static const uint32_t DeeployNetwork_input_20_len = 32;
extern int32_t *DeeployNetwork_input_21;
static const uint32_t DeeployNetwork_input_21_len = 32;
extern float32_t *DeeployNetwork_input_22;
static const uint32_t DeeployNetwork_input_22_len = 288;
extern float32_t *DeeployNetwork_input_23;
static const uint32_t DeeployNetwork_input_23_len = 9;
extern float32_t *DeeployNetwork_output_0;
static const uint32_t DeeployNetwork_output_0_len = 1.0;
extern float32_t *DeeployNetwork_output_1;
static const uint32_t DeeployNetwork_output_1_len = 9;
static const uint32_t DeeployNetwork_num_inputs = 24;
static const uint32_t DeeployNetwork_num_outputs = 2;
extern void *DeeployNetwork_inputs[24];
extern void *DeeployNetwork_outputs[2];
static const uint32_t DeeployNetwork_inputs_bytes[24] = {39200, 8,    32,   32,   64,   64, 64, 64, 128, 128, 128,  128,
                                                         32,    2048, 2048, 3584, 7168, 32, 64, 64, 128, 128, 1152, 36};
static const uint32_t DeeployNetwork_outputs_bytes[2] = {4.0, 36};
#endif
