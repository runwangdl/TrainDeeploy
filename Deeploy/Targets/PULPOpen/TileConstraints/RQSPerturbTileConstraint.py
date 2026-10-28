# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t, uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme, calculateFlatOffset, stridesFromShape


class RQSPerturbTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        mulBufferName = parseDict['mul']
        outputBufferName = parseDict['data_out']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, mulBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputShape = ctxt.lookup(inputBufferName).shape
        outputShape = ctxt.lookup(outputBufferName).shape

        mulBufferShapeLen = len(ctxt.lookup(mulBufferName).shape)

        # QW: this op is ELEMENTWISE -- input and output hold the same elements -- but their RANKS
        #     can differ. A perturbed requant bias is rank-1 `(cout,)`, and when a downstream
        #     consumer needs it broadcast (which the un-merged RequantShift does) Deeploy rewrites
        #     the OUTPUT tensor's shape to `(1, cout)` while the node's input stays `(cout,)`.
        #
        #     Pairing dims by raw index then equates `data_in.dim0 (cout)` with
        #     `data_out.dim0 (1)`, so the tiler sized the INPUT tile at ONE ELEMENT -- 4 bytes --
        #     while `serializeTilingSolution` still scheduled the full 128-byte DMA. Observed
        #     directly in the generated C: `data_in` at arena+128 and `mul` at arena+132, four bytes
        #     apart, each receiving a 128-byte transfer. The bias DMA clobbered the multiplier array
        #     and the perturbation came out wrong by a constant per output channel (channel 0 right,
        #     channels 1..31 wrong -- 1240 of 1280 outputs on block 3).
        #
        #     Align the axes the way numpy broadcasting does -- from the first non-unit axis onward --
        #     so ranks may differ as long as the real axes correspond. A weight `(cout, cin, H, W)`
        #     is unaffected (no leading unit axis on either side). -- QW
        def _firstNonUnit(shape) -> int:
            idx = 0
            while idx < len(shape) - 1 and shape[idx] == 1:
                idx += 1
            return idx

        inOff, outOff = _firstNonUnit(inputShape), _firstNonUnit(outputShape)

        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = 0)

        # Channel dim is the first REAL (non-unit) axis, since we manipulate parameter tensors.
        inChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = inOff)

        tilerModel.addConstraint(mulChannelVar == inChannelVar)

        for dim in range(min(len(inputShape) - inOff, len(outputShape) - outOff)):
            inputDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = inOff + dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = outOff + dim)
            tilerModel.addConstraint(inputDimVar == outputDimVar)  # Channel dim

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'mul', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        inputCubes = outputCubes

        rqCubes = []

        replacements = {"size": [], "channel_width": [], "tile_seed_offset": []}
        replacementTypes = {
            "size": PointerClass(uint16_t),
            "channel_width": PointerClass(uint16_t),
            "tile_seed_offset": PointerClass(uint32_t),  # uint32: global offsets exceed uint16
        }

        # Per-tile global element offset, so the RNG seed differs between tiles.
        outShape = ctxt.lookup(operatorRepresentation['data_out']).shape
        if isinstance(outShape, int):  # 1-D buffers store shape as a bare int
            outShape = (outShape,)
        outStrides = stridesFromShape(outShape)
        for absCube in absoluteOutputCubes:
            replacements['tile_seed_offset'].append(calculateFlatOffset(absCube.absoluteOffset, outStrides))

        for cube in inputCubes:
            
            # this OP is applied to weights, the output channels are the dim that matters, ant that's always the first.
            # QW: ...but skip LEADING UNIT axes first. A perturbed requant bias is rank-1 `(cout,)`
            #     until something downstream needs it broadcast, at which point Deeploy rewrites the
            #     SHARED buffer's shape to `(1, cout)` -- and then `dims[0]` is 1, giving
            #     `channel_width = size // 1 = cout` instead of 1. The kernel indexes
            #     `M[(start_offset + i) / channel_width]` (RandomNoiseQuant.c:182), so every element
            #     would take `M[0]`: one multiplier for all output channels.
            #
            #     This never fired while every conv was fused into a RequantizedConv, because then
            #     the bias is consumed by the fused node and never broadcast. NE16 leaves the
            #     RequantShift standalone (streamin forces int32 out), the broadcast happens, and the
            #     perturbation silently used the wrong per-channel multipliers -- 40 of 1280 outputs
            #     wrong on block 3, i.e. exactly one output channel.
            #
            #     A leading axis of extent 1 carries no channels, so skipping it is right for both
            #     shapes: `(1, cout)` -> cout, and a weight `(cout, cin, H, W)` -> cout (unchanged). -- QW
            chIdx = 0
            while chIdx < len(cube.dims) - 1 and cube.dims[chIdx] == 1:
                chIdx += 1
            rqCube = HyperRectangle((cube.offset[chIdx],), (cube.dims[chIdx],))
            channelDim = cube.dims[chIdx]

            rqCubes.append(rqCube)

            size = np.prod(cube.dims)
            channelWidth = size // channelDim
            channels = channelDim

            replacements['size'].append(size)
            replacements['channel_width'].append(channelWidth)
            # always channel first so we only need channel_width
            #replacements['channels'].append(channels) 

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, rq in zip(inputCubes, rqCubes):
            inputLoadSchedule.append({"data_in": a, "mul": rq})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
