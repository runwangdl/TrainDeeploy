# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Ported verbatim from Deeploy zo-support (FP32 PerturbRademacher ZO op). -- QW

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatPerturbRademacherTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Add the node's unique ID to help create a unique seed.
        operatorRepresentation['node_id'] = operatorRepresentation['nodeIdx']
        # Per-tile seed offset (overridden per-tile when tiled; 0 when not tiled)
        operatorRepresentation['tile_seed_offset'] = 0
        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatPerturbRademacherTemplate("""
// PerturbRademacher (Name: ${nodeName}, Op: ${nodeOp})
uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = (uint32_t) MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = (uint32_t) MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});
uint32_t ${nodeName}_local_size = ${nodeName}_chunk_stop - ${nodeName}_chunk_start;

// ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
// (0 / off) reproduce the baked ${seed}/${eps} behavior; the ZO runner sets them per pass. -- QW
uint32_t chunk_seed = ((${seed} + perturb_seed_base) + NUM_CORES * ${node_id} + ${nodeName}_core_id) ^ (${tile_seed_offset} * 0x9E3779B1u);
float32_t ${nodeName}_eps = perturb_eps_use_override ? perturb_eps_override : ${eps}f;
// RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
if (perturb_eps_use_override && zo_eco_on) {
ApplyECORademacher_fp32((const float32_t *) &${data_in}[${nodeName}_chunk_start], (float32_t *) &${data_out}[${nodeName}_chunk_start],
                        chunk_seed, perturbation_sign, ${nodeName}_local_size, zo_eco_mom[${node_id}] + ${nodeName}_chunk_start);
} else {
#ifdef USE_SEQUENTIAL_RADEMACHER
ApplySequentialRademacherPerturbation((const float32_t *)  &${data_in}[${nodeName}_chunk_start],
                                (float32_t *) &${data_out}[${nodeName}_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                ${nodeName}_local_size,
                                ${nodeName}_eps);
#else
ApplyRademacherPerturbation((const float32_t *)  &${data_in}[${nodeName}_chunk_start],
                                (float32_t *) &${data_out}[${nodeName}_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                ${nodeName}_local_size,
                                ${nodeName}_eps);
#endif
}

""")