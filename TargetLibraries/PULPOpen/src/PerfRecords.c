/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp6): storage for the --profileMicrobenchmark records (see perf_utils.h).
 * One definition, shared by Network.c (writer, cluster core 0) and the harness (reader, GAP9 FC).
 */
#include "perf_utils.h"

perf_record_t perf_records[PERF_MAX_RECORDS];
uint32_t perf_record_count = 0;
