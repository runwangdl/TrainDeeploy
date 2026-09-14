/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _MCHAN_H
#define _MCHAN_H

// Requires to have MCHAN_BASE_ADDR, MCHAN_EVENT defined outside of header
#ifndef MCHAN_BASE_ADDR
#error "[mchan.h] MCHAN_BASE_ADDR not defined!"
#endif

#if !defined(MCHAN_EVENT) && !defined(MCHAN_POLLED)
#error "[mchan.h] Nor MCHAN_EVENT nor MCHAN_POLLED defined!"
#endif

#if defined(MCHAN_EVENT) && !defined(MCHAN_EVENT_BIT)
#error                                                                         \
    "[mchan.h] MCHAN_EVENT_BIT should be defined when using events as signalization!"
#endif

#include "pmsis.h"

#define MCHAN_CMD_OFFSET 0
#define MCHAN_STATUS_OFFSET 4

#define MCHAN_CMD_ADDR (MCHAN_BASE_ADDR + MCHAN_CMD_OFFSET)
#define MCHAN_STATUS_ADDR (MCHAN_BASE_ADDR + MCHAN_STATUS_OFFSET)

#define READ_REG(addr) (*(volatile int *)(addr))
#define WRITE_REG(addr, value)                                                 \
  do {                                                                         \
    *(volatile int *)(addr) = (int)value;                                      \
  } while (0)

#define MCHAN_READ_CMD() READ_REG(MCHAN_CMD_ADDR)
#define MCHAN_WRITE_CMD(value) WRITE_REG(MCHAN_CMD_ADDR, value)

#define MCHAN_READ_STATUS() READ_REG(MCHAN_STATUS_ADDR)
#define MCHAN_WRITE_STATUS(value) WRITE_REG(MCHAN_STATUS_ADDR, value)

// MCHAN version 7 has 1 more bit for the transfer length, so all the flag
// offsets are shifted by 1. Also, LOC (TCDM) striding is not supported in v6.
#if MCHAN_VERSION == 7
#define MCHAN_TRANSFER_LEN_SIZE (17)
#else
#define MCHAN_TRANSFER_LEN_SIZE (16)
#endif

#define MCHAN_CMD_FLAG_DIRECTION_LOC2EXT (0 << (MCHAN_TRANSFER_LEN_SIZE + 0))
#define MCHAN_CMD_FLAG_DIRECTION_EXT2LOC (1 << (MCHAN_TRANSFER_LEN_SIZE + 0))
#define MCHAN_CMD_FLAG_INCREMENTAL (1 << (MCHAN_TRANSFER_LEN_SIZE + 1))
#define MCHAN_CMD_FLAG_2D_TRANSFER_EXTERNAL (1 << (MCHAN_TRANSFER_LEN_SIZE + 2))
#define MCHAN_CMD_FLAG_EVENT_ENABLE (1 << (MCHAN_TRANSFER_LEN_SIZE + 3))
#define MCHAN_CMD_FLAG_INTERRUPT_ENABLE (1 << (MCHAN_TRANSFER_LEN_SIZE + 4))
#define MCHAN_CMD_FLAG_BROADCAST_FINISH (1 << (MCHAN_TRANSFER_LEN_SIZE + 5))
#if MCHAN_VERSION == 7
#define MCHAN_CMD_FLAG_2D_TRANSFER_LOCAL                                       \
  (1 << (MCHAN_TRANSFER_LEN_SIZE + 6)) // can only be used with MCHAN v7
#endif
#define MCHAN_CMD_SHIFT_DIRECTION MCHAN_TRANSFER_LEN_SIZE

#define MCHAN_CMD(len, dir, inc, loc_2d, ext_2d, int_en, event_en, broadcast)  \
  (len | dir | inc | loc_2d | ext_2d | broadcast | int_en | event_en)

typedef enum {
  MCHAN_DMA_TRANSFER_DIRECTION_EXT2LOC = MCHAN_CMD_FLAG_DIRECTION_EXT2LOC,
  MCHAN_DMA_TRANSFER_DIRECTION_LOC2EXT = MCHAN_CMD_FLAG_DIRECTION_LOC2EXT
} mchan_dma_transfer_direction_e;

typedef struct {
  int cmd;
  int size;

  void *loc;
  int loc_size_1d;
  int loc_stride_1d;

  void *ext;
  int ext_size_1d;
  int ext_stride_1d;
} mchan_transfer_t;

#ifdef DEEPLOY_L1_AS_L2

/* Untiled baseline. With DEEPLOY_L1_AS_L2 the "L1" buffers are allocated out of
 * FC L2 (see gap9L1AllocateTemplate), so source and destination are both L2 and
 * there is nothing for the DMA to do. It cannot be asked to do it either: the
 * mchan hardware ignores the destination address and routes `loc` into the
 * cluster L1 banks through the low bits, so a real transfer would write into L1
 * and leave the L2 destination untouched -- the out-of-range L1 bank requests
 * and computed=0.0 that PR #21 first ran into.
 *
 * Replace the transfers with memcpy and make the channel API inert. Unlike
 * PR #21 the 2D variant is provided as well, so a model whose tiler emits
 * strided transfers is handled rather than silently skipped.
 *
 * The cost this measures is real: the cluster reaches these buffers over the L2
 * fabric instead of out of 1-cycle TCDM. */

#include <string.h>

static int mchan_transfer_get_id() { return 0; }

static void mchan_transfer_push_1d(mchan_transfer_t trans) {
  uint32_t size = (uint32_t)trans.cmd & ((1u << MCHAN_TRANSFER_LEN_SIZE) - 1);
  if (trans.cmd & MCHAN_CMD_FLAG_DIRECTION_EXT2LOC) {
    memcpy(trans.loc, trans.ext, size);
  } else {
    memcpy(trans.ext, trans.loc, size);
  }
}

static void mchan_transfer_push_2d(mchan_transfer_t trans) {
  uint32_t total = (uint32_t)trans.cmd & ((1u << MCHAN_TRANSFER_LEN_SIZE) - 1);
  uint32_t line = (uint32_t)trans.ext_size_1d;
  uint32_t stride = (uint32_t)trans.ext_stride_1d;
  int ext2loc = trans.cmd & MCHAN_CMD_FLAG_DIRECTION_EXT2LOC;
  char *loc = (char *)trans.loc;
  char *ext = (char *)trans.ext;
  if (line == 0) {
    return;
  }
  for (uint32_t moved = 0; moved < total; moved += line) {
    uint32_t n = (total - moved < line) ? (total - moved) : line;
    if (ext2loc) {
      memcpy(loc, ext, n);
    } else {
      memcpy(ext, loc, n);
    }
    loc += n;      /* local side is contiguous */
    ext += stride; /* external side is strided */
  }
}

static void mchan_transfer_push(mchan_transfer_t trans) {
  if (trans.ext_size_1d < trans.size) {
    mchan_transfer_push_2d(trans);
  } else {
    mchan_transfer_push_1d(trans);
  }
}

static void mchan_transfer_wait(int id) { (void)id; }
static void mchan_transfer_free(int id) { (void)id; }

#else

static int mchan_transfer_get_id() { return MCHAN_READ_CMD(); }

static void mchan_transfer_push_1d(mchan_transfer_t trans) {
  MCHAN_WRITE_CMD(trans.cmd);
  MCHAN_WRITE_CMD(trans.loc);
  MCHAN_WRITE_CMD(trans.ext);
}

static void mchan_transfer_push_2d(mchan_transfer_t trans) {
  MCHAN_WRITE_CMD(trans.cmd);
  MCHAN_WRITE_CMD(trans.loc);
  MCHAN_WRITE_CMD(trans.ext);
// MCHAN version 7 takes 2D "count" (length of 1D transfers) and stride in 2
// steps, v7 takes it in 1 step with the stride shifted to the upper 16 bits.
#if MCHAN_VERSION == 7
  MCHAN_WRITE_CMD(trans.ext_size_1d);
  MCHAN_WRITE_CMD(trans.ext_stride_1d);
#else
  MCHAN_WRITE_CMD(trans.ext_size_1d | (trans.ext_stride_1d << 16));
#endif
}

static void mchan_transfer_push(mchan_transfer_t trans) {
  MCHAN_WRITE_CMD(trans.cmd);
  MCHAN_WRITE_CMD(trans.loc);
  MCHAN_WRITE_CMD(trans.ext);

  if (trans.ext_size_1d < trans.size) {
    MCHAN_WRITE_CMD(trans.ext_size_1d);
    MCHAN_WRITE_CMD(trans.ext_stride_1d);
  }

  if (trans.loc_size_1d < trans.size) {
    MCHAN_WRITE_CMD(trans.loc_size_1d);
    MCHAN_WRITE_CMD(trans.loc_stride_1d);
  }
}

static void mchan_transfer_free(int tid) { MCHAN_WRITE_STATUS(1 << tid); }

static int mchan_transfer_busy(int tid) {
  return MCHAN_READ_STATUS() & (1 << tid);
}

static void mchan_transfer_wait(int tid) {
#if defined(MCHAN_EVENT)
  while (mchan_transfer_busy(tid))
    eu_evt_maskWaitAndClr(1 << MCHAN_EVENT_BIT);
#elif defined(MCHAN_POLLED)
  while (mchan_transfer_busy(tid))
    ;
#endif
}

#endif /* DEEPLOY_L1_AS_L2 */

#endif
