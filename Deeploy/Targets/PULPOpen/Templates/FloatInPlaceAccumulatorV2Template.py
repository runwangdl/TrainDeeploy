# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _PULPInPlaceAccumulatorV2Template(NodeTemplate):
    """True in-place InPlaceAccumulatorV2 template for PULP.

    Writes the accumulation result into ``accum_buffer`` (the graph input).
    ``data_out`` is registered as an alias of ``accum_buffer`` so the memory
    allocator knows they share memory and will not free ``accum_buffer``
    prematurely.

    ``data_out`` is intentionally *not* written by the emitted C code:

    - InPlaceAccumulatorV2 is terminal in the training graph — no downstream
      kernel consumes ``data_out``; it only exists as a symbolic output so
      the graph stays well-formed.
    - In the tiled path, emitting a write to ``data_out`` would also make
      Deeploy generate an L2 egress DMA for it, and ``data_out``'s L2 slot
      may overlap with other live buffers, corrupting L2.

    Semantics:
        if lazy_reset_grad: accum_buffer  = gradient   (reset)
        else:               accum_buffer += gradient   (accumulate)
    """

    def alignToContext(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        accum_buffer = ctxt.lookup(operatorRepresentation['accum_buffer'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        accum_buffer.aliases.add(data_out.name)
        data_out.aliases.add(accum_buffer.name)
        data_out._alias = accum_buffer.name

        return ctxt, operatorRepresentation, []


referenceTemplate = _PULPInPlaceAccumulatorV2Template("""
// InPlaceAccumulatorV2 (Name: ${nodeName}, Op: ${nodeOp})
// Writes result into accum_buffer (in-place).  data_out is an alias of
// accum_buffer and is deliberately not written — it has no downstream
// consumer, and emitting a write would trigger an L2 egress DMA whose
// destination may overlap with live buffers in the tiled path.
// Reset (lazy_reset_grad=1): accum_buffer  = gradient
// Accum (lazy_reset_grad=0): accum_buffer += gradient
//
// 6x unrolled, mirroring SGDTemplate. A plain `acc[i] += grad[i]` loop
// serialises on the FP-add latency: the next iteration's load-add-store waits
// on the previous write-back, and with nothing else in flight the pipeline
// stalls every iteration. Splitting the unrolled body into an add phase and a
// store phase gives the scheduler six independent adds to overlap. Measured on
// the MLperf AD autoencoder (81,920-element weight gradients, 9 tiles): 9.70 ->
// (SGD on the identical tensors, already unrolled, runs at 2.47 cyc/element).
// This matters because accumulation cost scales with PARAMETER COUNT while the
// backward scales with MACs, so it dominates parameter-dense, low-MAC models
// (55.5% of the autoencoder's training cycles before this change).
int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
int32_t ${nodeName}_start = MIN(${nodeName}_chunk * ${nodeName}_core_id, (int32_t)${size});
int32_t ${nodeName}_stop  = MIN(${nodeName}_start + ${nodeName}_chunk,   (int32_t)${size});

float32_t *${nodeName}_acc  = ${accum_buffer};
float32_t *${nodeName}_grad = ${gradient};
float32_t ${nodeName}_tmp[6];

int32_t ${nodeName}_i = ${nodeName}_start;

if (${lazy_reset_grad}[0]) {
    for (; ${nodeName}_i + 5 < ${nodeName}_stop; ${nodeName}_i += 6) {
        ${nodeName}_acc[${nodeName}_i]     = ${nodeName}_grad[${nodeName}_i];
        ${nodeName}_acc[${nodeName}_i + 1] = ${nodeName}_grad[${nodeName}_i + 1];
        ${nodeName}_acc[${nodeName}_i + 2] = ${nodeName}_grad[${nodeName}_i + 2];
        ${nodeName}_acc[${nodeName}_i + 3] = ${nodeName}_grad[${nodeName}_i + 3];
        ${nodeName}_acc[${nodeName}_i + 4] = ${nodeName}_grad[${nodeName}_i + 4];
        ${nodeName}_acc[${nodeName}_i + 5] = ${nodeName}_grad[${nodeName}_i + 5];
    }
    for (; ${nodeName}_i < ${nodeName}_stop; ${nodeName}_i++) {
        ${nodeName}_acc[${nodeName}_i] = ${nodeName}_grad[${nodeName}_i];
    }
} else {
    for (; ${nodeName}_i + 5 < ${nodeName}_stop; ${nodeName}_i += 6) {
        // add phase — six independent FP adds, no write-back dependency
        ${nodeName}_tmp[0] = ${nodeName}_acc[${nodeName}_i]     + ${nodeName}_grad[${nodeName}_i];
        ${nodeName}_tmp[1] = ${nodeName}_acc[${nodeName}_i + 1] + ${nodeName}_grad[${nodeName}_i + 1];
        ${nodeName}_tmp[2] = ${nodeName}_acc[${nodeName}_i + 2] + ${nodeName}_grad[${nodeName}_i + 2];
        ${nodeName}_tmp[3] = ${nodeName}_acc[${nodeName}_i + 3] + ${nodeName}_grad[${nodeName}_i + 3];
        ${nodeName}_tmp[4] = ${nodeName}_acc[${nodeName}_i + 4] + ${nodeName}_grad[${nodeName}_i + 4];
        ${nodeName}_tmp[5] = ${nodeName}_acc[${nodeName}_i + 5] + ${nodeName}_grad[${nodeName}_i + 5];
        // store phase
        ${nodeName}_acc[${nodeName}_i]     = ${nodeName}_tmp[0];
        ${nodeName}_acc[${nodeName}_i + 1] = ${nodeName}_tmp[1];
        ${nodeName}_acc[${nodeName}_i + 2] = ${nodeName}_tmp[2];
        ${nodeName}_acc[${nodeName}_i + 3] = ${nodeName}_tmp[3];
        ${nodeName}_acc[${nodeName}_i + 4] = ${nodeName}_tmp[4];
        ${nodeName}_acc[${nodeName}_i + 5] = ${nodeName}_tmp[5];
    }
    for (; ${nodeName}_i < ${nodeName}_stop; ${nodeName}_i++) {
        ${nodeName}_acc[${nodeName}_i] += ${nodeName}_grad[${nodeName}_i];
    }
}
""")

# Single-core variant: full range on the calling core (no pi_core_id chunking).
# Bound on GAP9 with the non-fork transformer: dense back-to-back accumulator
# forks lock up the GVSoC EU dispatch unit; accumulation is cheap & elementwise.
singleCoreTemplate = _PULPInPlaceAccumulatorV2Template("""
// InPlaceAccumulatorV2 single-core (Name: ${nodeName}, Op: ${nodeOp})
// Same 6x unrolling as the parallel variant — the FP-add latency stall it
// hides is per-core, so it applies here too.
float32_t *${nodeName}_acc  = ${accum_buffer};
float32_t *${nodeName}_grad = ${gradient};
float32_t ${nodeName}_tmp[6];
int32_t ${nodeName}_n = (int32_t)${size};
int32_t ${nodeName}_i = 0;

if (${lazy_reset_grad}[0]) {
    for (; ${nodeName}_i + 5 < ${nodeName}_n; ${nodeName}_i += 6) {
        ${nodeName}_acc[${nodeName}_i]     = ${nodeName}_grad[${nodeName}_i];
        ${nodeName}_acc[${nodeName}_i + 1] = ${nodeName}_grad[${nodeName}_i + 1];
        ${nodeName}_acc[${nodeName}_i + 2] = ${nodeName}_grad[${nodeName}_i + 2];
        ${nodeName}_acc[${nodeName}_i + 3] = ${nodeName}_grad[${nodeName}_i + 3];
        ${nodeName}_acc[${nodeName}_i + 4] = ${nodeName}_grad[${nodeName}_i + 4];
        ${nodeName}_acc[${nodeName}_i + 5] = ${nodeName}_grad[${nodeName}_i + 5];
    }
    for (; ${nodeName}_i < ${nodeName}_n; ${nodeName}_i++) {
        ${nodeName}_acc[${nodeName}_i] = ${nodeName}_grad[${nodeName}_i];
    }
} else {
    for (; ${nodeName}_i + 5 < ${nodeName}_n; ${nodeName}_i += 6) {
        ${nodeName}_tmp[0] = ${nodeName}_acc[${nodeName}_i]     + ${nodeName}_grad[${nodeName}_i];
        ${nodeName}_tmp[1] = ${nodeName}_acc[${nodeName}_i + 1] + ${nodeName}_grad[${nodeName}_i + 1];
        ${nodeName}_tmp[2] = ${nodeName}_acc[${nodeName}_i + 2] + ${nodeName}_grad[${nodeName}_i + 2];
        ${nodeName}_tmp[3] = ${nodeName}_acc[${nodeName}_i + 3] + ${nodeName}_grad[${nodeName}_i + 3];
        ${nodeName}_tmp[4] = ${nodeName}_acc[${nodeName}_i + 4] + ${nodeName}_grad[${nodeName}_i + 4];
        ${nodeName}_tmp[5] = ${nodeName}_acc[${nodeName}_i + 5] + ${nodeName}_grad[${nodeName}_i + 5];
        ${nodeName}_acc[${nodeName}_i]     = ${nodeName}_tmp[0];
        ${nodeName}_acc[${nodeName}_i + 1] = ${nodeName}_tmp[1];
        ${nodeName}_acc[${nodeName}_i + 2] = ${nodeName}_tmp[2];
        ${nodeName}_acc[${nodeName}_i + 3] = ${nodeName}_tmp[3];
        ${nodeName}_acc[${nodeName}_i + 4] = ${nodeName}_tmp[4];
        ${nodeName}_acc[${nodeName}_i + 5] = ${nodeName}_tmp[5];
    }
    for (; ${nodeName}_i < ${nodeName}_n; ${nodeName}_i++) {
        ${nodeName}_acc[${nodeName}_i] += ${nodeName}_grad[${nodeName}_i];
    }
}
""")
