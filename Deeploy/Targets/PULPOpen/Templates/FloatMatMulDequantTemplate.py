# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Matmul whose weight operand is int8 and is dequantised inside the kernel.

A weight-only-quantised model stores the frozen weight as int8 and needs it as
float for the fp32 matmul. Emitting that conversion as a separate Dequant node
materialises the whole dequantised matrix: on CCT-QLoRA, 1088 KB of fp32 tensors
that exist only to be read by the very next node, and that then have to be held
live from the forward pass to the backward one or recomputed. Profiling puts the
cost at 239.95M cycles for the Dequant nodes alone, of which only 48.47M is the
kernel -- the rest is moving those tensors.

QLoRA does not do this. It dequantises inside the matmul and discards the value,
which is what this template binds: PULP_MatMul_fp32_i8_fp32_unroll1x7 takes the
int8 weight plus the per-tensor scale and zero point and applies them in the loop.
"""

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Matmul with an int8 weight, dequantised in-kernel (Name: ${nodeName}, Op: ${nodeOp})

for(uint32_t b=0; b<${batch}; b++) {
    % if A_batched:
    ${A_type.typeName} batch_A = ${A} + b * ${M} * ${N};
    % else:
    ${A_type.typeName} batch_A = ${A};
    % endif

    % if B_batched:
    ${B_type.typeName} batch_B = ${B} + b * ${N} * ${O};
    % else:
    ${B_type.typeName} batch_B = ${B};
    % endif

    ${data_out_type.typeName} batch_out = ${data_out} + b * ${M} * ${O};

    PULP_MatMul_fp32_i8_fp32_unroll1x7(
        batch_A,
        batch_B,
        batch_out,
        ${M},
        ${N},
        ${O},
        ${dequant_scale}f,
        ${dequant_zero_point}
    );
}
""")
