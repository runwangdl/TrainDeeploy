# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


def _isBroadcast(shape) -> bool:
    return len([dim for dim in shape if dim != 1]) <= 1


class PULPFloatAddTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # An operand with a single non-unit axis is a vector added to every row of
        # the other. Keeping it that way avoids storing the same row once per output
        # row; the loop below reads it with a wrapped index instead.
        #
        # Either operand can be the broadcast one. A Linear exported as MatMul + Add
        # puts the bias second, but the same graph after lowering can present it
        # first, and indexing a two-element bias by the output index reads past it.
        # Normalise here so the template always adds data_in_2 into data_in_1.
        shape1 = ctxt.lookup(operatorRepresentation['data_in_1']).shape
        shape2 = ctxt.lookup(operatorRepresentation['data_in_2']).shape

        if _isBroadcast(shape2) and not _isBroadcast(shape1):
            broadcastShape = shape2
        elif _isBroadcast(shape1) and not _isBroadcast(shape2):
            operatorRepresentation['data_in_1'], operatorRepresentation['data_in_2'] = \
                operatorRepresentation['data_in_2'], operatorRepresentation['data_in_1']
            broadcastShape = shape1
        else:
            broadcastShape = None

        operatorRepresentation['broadcast'] = broadcastShape is not None
        if 'rowLen' not in operatorRepresentation:
            operatorRepresentation['rowLen'] = (broadcastShape[-1] if broadcastShape else (shape2[-1] if shape2 else 1))
        return ctxt, operatorRepresentation, []


referenceTemplate = PULPFloatAddTemplate("""
// Add Parallel with 1x6 unrolling (Name: ${nodeName}, Op: ${nodeOp})
uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = (uint32_t) MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = (uint32_t) MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});

% if broadcast:
// data_in_2 holds one row of ${rowLen} shared by every output row. Walk the row
// index alongside i rather than dividing: there is no hardware divide here.
uint32_t ${nodeName}_j = ${nodeName}_chunk_start % ${rowLen};
for (uint32_t i = ${nodeName}_chunk_start; i < ${nodeName}_chunk_stop; i++) {
    ${data_out}[i] = ${data_in_1}[i] + ${data_in_2}[${nodeName}_j];
    ${nodeName}_j++;
    if (${nodeName}_j == ${rowLen}) {
        ${nodeName}_j = 0;
    }
}
% else:
uint32_t i = ${nodeName}_chunk_start;
for (; i + 5 < ${nodeName}_chunk_stop; i += 6) {
    ${data_out}[i] = ${data_in_1}[i] + ${data_in_2}[i];
    ${data_out}[i+1] = ${data_in_1}[i+1] + ${data_in_2}[i+1];
    ${data_out}[i+2] = ${data_in_1}[i+2] + ${data_in_2}[i+2];
    ${data_out}[i+3] = ${data_in_1}[i+3] + ${data_in_2}[i+3];
    ${data_out}[i+4] = ${data_in_1}[i+4] + ${data_in_2}[i+4];
    ${data_out}[i+5] = ${data_in_1}[i+5] + ${data_in_2}[i+5];
}

for (; i < ${nodeName}_chunk_stop; i++) {
    ${data_out}[i] = ${data_in_1}[i] + ${data_in_2}[i];
}
% endif
""")
