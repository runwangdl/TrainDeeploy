# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _RQSPerturbRademacher_i32_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        operatorRepresentation['node_id'] = operatorRepresentation['nodeIdx']
        # Per-tile seed offset (overridden per-tile when tiled; 0 when not tiled)
        operatorRepresentation['tile_seed_offset'] = 0
        return ctxt, operatorRepresentation, []


referenceTemplate = _RQSPerturbRademacher_i32_Template("""
// PerturbRademacher_i32 (Name: ${nodeName}, Op: ${nodeOp})
uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);

// Parallelize over the total size of the tensor
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = (uint32_t) MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = (uint32_t) MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});
uint32_t ${nodeName}_local_size = ${nodeName}_chunk_stop - ${nodeName}_chunk_start;

// Calculate the starting channel for this core's chunk of M
uint32_t ${nodeName}_channel_start_offset = ${nodeName}_chunk_start % ${channel_width};

// Pick large enough stride to minimize correlation between nodes.
// -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
uint32_t chunk_seed = ((${seed} + perturb_seed_base) + NUM_CORES * ${node_id} + ${nodeName}_core_id) ^ (${tile_seed_offset} * 0x9E3779B1u);
// QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
// train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
float32_t ${nodeName}_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D
%>
// RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
// the update tensors are single-tile, so chunk_start is the element offset into it.
if (perturb_eps_use_override && zo_eco_on) {
ApplyECOQuantRademacher_i32((const int32_t *) &${data_in}[${nodeName}_chunk_start], (int32_t *) &${data_out}[${nodeName}_chunk_start],
                            (const int32_t *) ${mul}, ${log2Dstring}, ${channel_width}, chunk_seed, ${nodeName}_local_size,
                            ${nodeName}_chunk_start, perturbation_sign, zo_eco_mom[${node_id}] + ${nodeName}_chunk_start);
} else {
ApplyPerturbQuantRademacher_i32((const int32_t *)  &${data_in}[${nodeName}_chunk_start],
                                (int32_t *) &${data_out}[${nodeName}_chunk_start],
                                (const int32_t *) ${mul},
                                ${log2Dstring},
                                ${channel_width},
                                chunk_seed,
                                ${nodeName}_local_size,
                                ${nodeName}_chunk_start,
                                perturbation_sign,  // -- QW: +eps (L+) / -eps (L-)
                                ${nodeName}_eps_scale);
}  // -- QW: zo_update coeff scaling (1.0f = train pass)

""")
