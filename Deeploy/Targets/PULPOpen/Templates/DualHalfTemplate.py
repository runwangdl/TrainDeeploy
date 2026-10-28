# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp8): `DualHalf` -- copy half `half` (0 = + pass, 1 = - pass) of a channel-stacked vector
[rows, 2*C] into [rows, C]. The dual-pass tail gives each pass's Gemm exactly the unfused [1, C] input: Deeploy's tiled
Slice mis-strides (its tile constraint replaces data_in_size by the tile size), and zero-padding the FC weight to 2*C changes
the fp32 summation grouping of PULP_Gemm (blocks of six along K) -- not bit-exact (GAP9 step-1 L- differed)."""
from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// DualHalf (Name: ${nodeName}, Op: ${nodeOp}, half ${half}) -- QW exp8
// contiguous per-core chunk: the strided form (c = core_id; c += NUM_CORES) lost the last iteration of cores 1..7 on
// Siracusa (LLVM hardware-loop trip count), GAP9 was fine
uint32_t ${nodeName}_chunk = (${cols}u + NUM_CORES - 1u) / NUM_CORES;
uint32_t ${nodeName}_start = MIN(${nodeName}_chunk * (uint32_t) pi_core_id(), ${cols}u);
uint32_t ${nodeName}_stop = MIN(${nodeName}_start + ${nodeName}_chunk, ${cols}u);
for (uint32_t r = 0; r < ${rows}u; r++) {
  for (uint32_t c = ${nodeName}_start; c < ${nodeName}_stop; c++)
    ((float32_t *) ${data_out})[r * ${cols}u + c] = ((const float32_t *) ${data_in})[r * 2u * ${cols}u + ${half}u * ${cols}u + c];
}
""")
