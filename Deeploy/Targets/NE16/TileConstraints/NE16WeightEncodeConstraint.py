# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (exp16c): tile constraint for `NE16WeightEncode` -- deliberately NOT tileable.

The bit-serial encoding maps a rank-4 int8 weight `(cout, cin, H, W)` onto a rank-3 packed buffer
`(taps*cout, cinMajor, 16)`. The two tensors have different ranks and no shared dimension: output
row `j*cout + co` gathers from a strided slice of the input, and each output BYTE mixes 8 input
elements. There is no cube-to-cube relation for the tiler to express.

It also does not need one. The encoded weights are small -- block 1's is 4 KB, the largest
SpeechNet conv's is 7 KB -- and the consuming conv's own constraint
(`NE161xKConstraint.addGeometricalConstraint`) already pins the whole encoded weight resident,
since the template addresses taps *inside* it by offset. So both tensors are pinned to their full
extent and the node runs once, untiled. -- QW
"""

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class NE16WeightEncodeTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        for name in (parseDict['data_in'], parseDict['data_out']):
            tilerModel.addTensorDimToModel(ctxt, name)
            for dimIdx in range(len(ctxt.lookup(name).shape)):
                var = tilerModel.getTensorDimVar(tensorName = name, dimIdx = dimIdx)
                tilerModel.addConstraint(var == var.Max())
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, ['data_in', 'data_out'])

        inShape = tuple(int(s) for s in ctxt.lookup(operatorRepresentation['data_in']).shape)
        inCube = HyperRectangle((0,) * len(inShape), inShape)

        inputLoadSchedule = [{"data_in": inCube} for _ in outputCubes]
        outputLoadSchedule = [{"data_out": cube} for cube in outputCubes]

        # Nothing varies per tile -- there is only ever one tile.
        return VariableReplacementScheme({}, {}), \
            TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)


class NE16SignedInputBiasTileConstraint(TileConstraint):
    """QW (exp16c phase 4): also untiled -- a per-output-channel reduction over the whole weight.

    Output channel `co` sums the entire `(cin, H, W)` slab of the weight, so any tiling of the
    weight would have to carry partial sums across tiles. The tensors are tiny (block 0's weight is
    32 bytes, its bias 8 int32) and the consumer's constraint pins the bias whole anyway. -- QW
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        for name in (parseDict['weight'], parseDict['mul'], parseDict['add'], parseDict['data_out']):
            tilerModel.addTensorDimToModel(ctxt, name)
            for dimIdx in range(len(ctxt.lookup(name).shape)):
                var = tilerModel.getTensorDimVar(tensorName = name, dimIdx = dimIdx)
                tilerModel.addConstraint(var == var.Max())
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation,
                                                                  ['weight', 'mul', 'add', 'data_out'])

        def _whole(key: str) -> HyperRectangle:
            shape = tuple(int(s) for s in ctxt.lookup(operatorRepresentation[key]).shape)
            return HyperRectangle((0,) * len(shape), shape)

        inputLoadSchedule = [{"weight": _whole('weight'), "mul": _whole('mul'), "add": _whole('add')}
                             for _ in outputCubes]
        outputLoadSchedule = [{"data_out": cube} for cube in outputCubes]
        return VariableReplacementScheme({}, {}), \
            TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
