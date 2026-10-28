# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, _Template

_tileHeader = NodeTemplate("""
const uint32_t coreId = pi_core_id();

% for i in range(numDims):
uint16_t dimLen_${i} = <%text>${</%text>${dimLenPtr[i]}<%text>}</%text>;\n
% endfor
""")

_tileForLoop = NodeTemplate("""
const uint32_t baseChunk = dimLen_${i} / NUM_CORES;
const uint32_t leftover = dimLen_${i} - baseChunk * NUM_CORES;
const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
for(uint32_t i_${i} = offset; i_${i} < offset + chunk; i_${i}++ ) {
""")

_forLoop = NodeTemplate("""
for(uint32_t i_${i} = 0; i_${i} < dimLen_${i} ; i_${i}++){
""")

# RW 2026-10-13: stride-based loops. The VLA access dst[..][..] = src[..][..] recomputes every index from the
# runtime tile dims (dimLen_* come from per-tile tables), and GCC does not strength-reduce it: the SpeechNet
# block-0 output transpose ran 2.7x slower for a 5+3 channel split than for a constant 4+4 split. Each loop
# level now adds its index times a precomputed stride to a running offset once.
_strideForLoop = NodeTemplate("""
for(uint32_t i_${i} = 0; i_${i} < dimLen_${i} ; i_${i}++){
const uint32_t offS_${k} = offS_${kp} + i_${i} * sIn_${i};
const uint32_t offD_${k} = offD_${kp} + i_${i} * sOut_${k};
""")

_strideTileForLoop = NodeTemplate("""
const uint32_t baseChunk_${k} = dimLen_${i} / NUM_CORES;
const uint32_t leftover_${k} = dimLen_${i} - baseChunk_${k} * NUM_CORES;
const uint32_t offset_${k} = baseChunk_${k} * coreId + (coreId < leftover_${k} ? coreId : leftover_${k});
const uint32_t chunk_${k} = coreId < leftover_${k} ? baseChunk_${k} + 1 : baseChunk_${k};
for(uint32_t i_${i} = offset_${k}; i_${i} < offset_${k} + chunk_${k}; i_${i}++ ) {
const uint32_t offS_${k} = offS_${kp} + i_${i} * sIn_${i};
const uint32_t offD_${k} = offD_${kp} + i_${i} * sOut_${k};
""")


class PULPTransposeTemplate(NodeTemplate):

    def __init__(self, templateStr: str):
        self._indirectTemplate = _Template(templateStr)
        self.subTemplates = {}
        self.subTemplateGenerators = {}

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        shapeStr = ""
        dimStr = ""
        accessStr = ""
        outAccessStr = ""
        outShapeStr = ""
        perm = operatorRepresentation['perm']
        data_in_shape = ctxt.lookup(operatorRepresentation['data_in']).shape
        data_out_shape = ctxt.lookup(operatorRepresentation['data_out']).shape

        for idx, i in enumerate(perm[:-1]):
            shapeStr += '[' + f"dimLen_{idx+1}" + ']'
            outShapeStr += '[' + f"dimLen_{perm[idx+1]}" + ']'

        for dim in data_in_shape:
            dimStr += '[' + str(dim) + ']'

        for idx, i in enumerate(perm):
            accessStr += '[i_' + str(idx) + ']'
            outAccessStr += '[i_' + str(i) + ']'

        fRep = operatorRepresentation.copy()

        fRep['shapeStr'] = shapeStr
        fRep['outShapeStr'] = outShapeStr
        fRep['outAccessStr'] = outAccessStr
        fRep['dimStr'] = dimStr
        fRep['accessStr'] = accessStr
        fRep['data_out_shape'] = data_out_shape

        # RW 2026-10-13: parallelise the LARGEST output dimension. The first dim >= 8 of the full shape is
        # usually the channel axis, which is exactly the axis the tiler cuts (SpeechNet block-0 output
        # transpose: 2..4 channels per tile -> 2..4 of 8 cores busy, the node was 47% of the fused NE16 run).
        # parallelDims = [idx for idx, dim in enumerate(data_out_shape) if dim >= 8]
        # if len(parallelDims) > 0:
        #     parallelDim = parallelDims[0]
        # else:
        #     parallelDim = data_out_shape.index(max(data_out_shape))
        parallelDim = data_out_shape.index(max(data_out_shape))

        forLoops = []
        dimLenPtrs = []
        n = len(perm)
        for idx, i in enumerate(perm):
            operatorRepresentation[f"dimLen_{idx}"] = data_in_shape[idx]
            dimLenPtrs.append(f"dimLen_{idx}")
            gen = _strideTileForLoop if idx == parallelDim else _strideForLoop
            forLoops.append(gen.generate({"i": i, "k": idx, "kp": "m1" if idx == 0 else idx - 1}))
        # input strides (row-major over input dims) and output strides (row-major over output dims)
        strides = "const uint32_t offS_m1 = 0;\nconst uint32_t offD_m1 = 0;\n"
        for d in range(n - 1, -1, -1):
            strides += f"const uint32_t sIn_{d} = " + ("1" if d == n - 1 else f"sIn_{d+1} * dimLen_{d+1}") + ";\n"
        for k in range(n - 1, -1, -1):
            strides += f"const uint32_t sOut_{k} = " + ("1" if k == n - 1 else f"sOut_{k+1} * dimLen_{perm[k+1]}") + ";\n"
        fRep['strides'] = strides
        fRep['lastK'] = n - 1

        fRep['forLoops'] = forLoops
        fRep['tileHeader'] = _tileHeader.generate({"numDims": len(perm), "dimLenPtr": dimLenPtrs})
        fRep['parallelDim'] = parallelDim

        self.template = _Template(self._indirectTemplate.render(**fRep))

        return ctxt, operatorRepresentation, []


referenceTemplate = PULPTransposeTemplate("""
// Transpose ${data_in_shape} -> ${data_out_shape} (Name: ${nodeName}, Op: ${nodeOp})
${tileHeader}
// RW: GCC Segmentation fault
// RW 2026-10-13: flat pointers + strides (was: VLA casts src${shapeStr} / dst${outShapeStr})
const ${data_in_type.referencedType.typeName} *src = (const ${data_in_type.referencedType.typeName} *)<%text>${data_in}</%text>;
${data_in_type.referencedType.typeName} *dst = (${data_in_type.referencedType.typeName} *)<%text>${data_out}</%text>;
${strides}
% for idx, i in enumerate(perm):
${forLoops[idx]}
% endfor
dst[offD_${lastK}] = src[offS_${lastK}];
% for idx, i in enumerate(perm):
}
% endfor
""")
