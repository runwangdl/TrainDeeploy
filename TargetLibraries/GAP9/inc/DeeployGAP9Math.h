/*
 * SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
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

#define BEGIN_SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0)

#include "DeeployBasicMath.h"

// RW: Declare the GAP9-native float kernels (defined in GAP9/src/). GAP9 has
// its own kernel implementations with signatures that differ from PULPOpen's,
// so it cannot include PULPOpen's kernel headers. Without these prototypes the
// generated TrainingNetwork.c called the high-arity gradient kernels (>8 args,
// hence stack-passed) with implicit (int) declarations, mismatching the ABI and
// corrupting memory after training (a gvsoc "not indexable" abort).
#include "kernel/GAP9Kernels.h"

#include "dory_dma.h"
#include "dory_mem.h"

#include "pmsis.h"

#endif // __DEEPLOY_MATH_HEADER_
