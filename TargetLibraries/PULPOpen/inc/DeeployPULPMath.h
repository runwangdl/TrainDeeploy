/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_HEADER_
#define __DEEPLOY_MATH_HEADER_

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "types.h"

#define BEGIN_SINGLE_CORE if (pi_core_id() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (pi_core_id() == 0)

#include "DeeployBasicMath.h"

#include "pmsis.h"

#include "kernel/BatchNorm.h"
#include "kernel/Conv.h"
#include "kernel/ConvGrad.h"
#include "kernel/DualConv.h" // -- QW (dual_pass_fusion exp5)
#include "kernel/FlatConv.h" // -- QW (dual_pass_fusion exp5a)
#include "kernel/Conv4x2.h"  // -- QW (dual_pass_fusion exp7a step 2a)
#include "kernel/DualConv2In.h" // -- QW (dual_pass_fusion exp7 step 3)
#include "kernel/PerturbConv4x2.h" // -- QW (exp11): single-pass in-kernel-perturbation conv
#include "kernel/Im2ColU8.h" // -- QW (dual_pass_fusion exp6)
#include "kernel/GELU.h"
#include "kernel/Layernorm.h"
#include "kernel/Matmul.h"
#include "kernel/MaxPool.h"
#include "kernel/RandomNoise.h" // -- QW
#include "kernel/ZORuntime.h" // -- QW (ZO runtime perturb controls)
#include "kernel/RQiHardswish.h"
#include "kernel/RequantShift.h"
#include "kernel/Softmax.h"
#include "kernel/UniformRequantShift.h"
#include "kernel/gemv.h"
#include "kernel/iRMSnorm.h"

#define LOG2(x) (__builtin_pulp_fl1(x))

#endif // __DEEPLOY_MATH_HEADER_
