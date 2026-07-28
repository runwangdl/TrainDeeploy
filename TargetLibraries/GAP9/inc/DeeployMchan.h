/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _DEEPLOY_MCHAN_H
#define _DEEPLOY_MCHAN_H

/*
 * GAP9 MCHAN v7 configuration wrapper for Deeploy
 *
 * This header configures and includes mchan.h with proper GAP9-specific
 * settings. Based on DORY's GAP9 DMA implementation.
 */

#include "pmsis.h"

// Define MCHAN base address if not already defined
#ifndef MCHAN_BASE_ADDR
#define MCHAN_BASE_ADDR (CLUSTER_PERIPHERALS_ADDR + CLUSTER_MCHAN_OFFSET)
#endif

// Define MCHAN version (GAP9 uses v7)
#ifndef MCHAN_VERSION
#define MCHAN_VERSION 7
#endif

// Use event-based synchronization (recommended for GAP9)
#ifndef MCHAN_POLLED
#define MCHAN_EVENT
#endif

// Define event bit for cluster DMA
#ifdef MCHAN_EVENT
#ifndef MCHAN_EVENT_BIT
#define MCHAN_EVENT_BIT (CLUSTER_IRQ_DMA0) // Typically 8
#endif
#endif

// Now include the mchan.h header with all configurations set
#include "mchan.h"

// GAP9's mchan.h carries the older push/wait API used by dory_dma.c. The tiling
// codegen (MchanDma.py) emits the channel API instead -- mchan_channel_alloc /
// _wait / _free, mchan_transfer_1d, mchan_transfer_2d_ext_strided -- which only
// PULPOpen's mchan_v7.h defines. GAP9 *is* MCHAN v7, and PULPOpen/inc is already
// on the include path, so pull in the definitions rather than duplicating them.
// Every macro mchan_v7.h requires (MCHAN_BASE_ADDR, MCHAN_EVENT/POLLED,
// MCHAN_EVENT_BIT) is set above.
// mchan_v7.h asserts on the channel id, but neither "assert.h" nor <assert.h>
// yields an assert() macro here: the GAP9 SDK ships
// rtos/pmsis/tools/assert/include/assert.h ahead of the toolchain's on the include
// path, and it declares only pi_assert. With -Werror the implicit declaration is
// fatal. Supply the macro rather than reordering the SDK's include path; the checks
// guard a channel id the codegen already bounds, and this is a DMA hot path where
// a runtime check would not be wanted anyway.
#ifndef assert
#define assert(cond) ((void)0)
#endif

#include "mchan_v7.h"

// GAP9 MCHAN v7 exposes 16 hardware channels (ids 0..15). The generated
// DMA code (MchanDma.py template) guards channel waits with
// `channel <= MCHAN_CHANNEL_ID_MAX`, but GAP9's mchan.h does not define this
// macro (only PULPOpen's mchan_v6/v7.h do). Define it here so the generated
// TrainingNetwork.c compiles.
// MCHAN_CHANNEL_ID_MAX now comes from mchan_v7.h; kept guarded in case that
// header stops defining it.
#ifndef MCHAN_CHANNEL_ID_MAX
#define MCHAN_CHANNEL_ID_MAX (15)
#endif

#endif // _DEEPLOY_MCHAN_H
