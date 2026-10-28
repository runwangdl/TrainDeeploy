
#ifndef __DEEPLOY_HEADER__
#define __DEEPLOY_HEADER__
#include "DeeployPULPMath.h"
#include "bsp/ram.h"
#include "dory_mem.h"
#include "mchan_siracusa.h"
#include "perf_utils.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "stdint.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void RunNetwork();
void InitNetwork();

extern int8_t *DeeployNetwork_MEMORYARENA_L1;
static const uint32_t DeeployNetwork_MEMORYARENA_L1_len = 124732;
extern int8_t *DeeployNetwork_MEMORYARENA_L2;
static const uint32_t DeeployNetwork_MEMORYARENA_L2_len = 157696;
extern float32_t *DeeployNetwork_input_0;
static const uint32_t DeeployNetwork_input_0_len = 9744;
extern float32_t *DeeployNetwork_input_1;
static const uint32_t DeeployNetwork_input_1_len = 2048;
extern float32_t *DeeployNetwork_input_2;
static const uint32_t DeeployNetwork_input_2_len = 16;
extern float32_t *DeeployNetwork_output_0;
static const uint32_t DeeployNetwork_output_0_len = 19712;
static const uint32_t DeeployNetwork_num_inputs = 3;
static const uint32_t DeeployNetwork_num_outputs = 1;
extern void *DeeployNetwork_inputs[3];
extern void *DeeployNetwork_outputs[1];
static const uint32_t DeeployNetwork_inputs_bytes[3] = {38976, 8192, 64};
static const uint32_t DeeployNetwork_outputs_bytes[1] = {78848};
#endif
