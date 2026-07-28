# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.Generic.TileConstraints.AddTileConstraint import AddTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


def isBroadcastOperand(shape) -> bool:
    """True when a tensor carries a single non-unit axis, i.e. one row repeated."""
    return len([dim for dim in shape if dim != 1]) <= 1


class PULPBroadcastAddTileConstraint(AddTileConstraint):
    """Add where the second operand may be a vector broadcast over the first.

    A Linear exported as MatMul + Add gives the Add a bias of shape [O] against an
    activation of shape [1, M, O]. ``AddLayer.computeShapes`` used to rewrite the
    bias to the activation's shape, which materialises the same O values M times:
    on CCT-2 at 64 tokens that is 32 KB per 128-wide bias holding 512 B of data,
    and the biases live at the peak account for 252 KB, around 13% of it.

    Keeping the bias at [O] means the two operands no longer have equal dimensions,
    which is the assumption ``BOPTileConstraint`` is built on, in two places:

    * the geometrical constraint ties every axis of both inputs to the output. For
      a broadcast operand only the trailing axis has anything to tie; the leading
      axes are 1 and stay 1 however the output is tiled.
    * the tiling schedule loads the same cube from both inputs. The broadcast
      operand instead needs the trailing slice of the output cube, since that is
      the part of the vector the tile actually reads.

    Both fall back to the inherited behaviour when the operands do have equal
    dimensions, so a genuine elementwise Add is unaffected.
    """

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBuffer2Shape = ctxt.lookup(parseDict[cls.dataIn2Name]).shape
        if not isBroadcastOperand(inputBuffer2Shape):
            return super().addGeometricalConstraint(tilerModel, parseDict, ctxt)

        inputBuffer1Name = parseDict[cls.dataIn1Name]
        inputBuffer2Name = parseDict[cls.dataIn2Name]
        outputBufferName = parseDict[cls.dataOutName]

        for bufferName in [inputBuffer1Name, inputBuffer2Name, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        input1Shape = ctxt.lookup(inputBuffer1Name).shape
        for dim in range(len(input1Shape)):
            inputDim1Var = tilerModel.getTensorDimVar(tensorName = inputBuffer1Name, dimIdx = dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)
            tilerModel.addConstraint(inputDim1Var == outputDimVar)

        # The broadcast operand contributes only its trailing axis, which must match
        # the output's trailing axis so a tile reads a whole row of it.
        broadcastDimVar = tilerModel.getTensorDimVar(tensorName = inputBuffer2Name,
                                                     dimIdx = len(inputBuffer2Shape) - 1)
        outputLastDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName,
                                                      dimIdx = len(input1Shape) - 1)
        tilerModel.addConstraint(broadcastDimVar == outputLastDimVar)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        inputBuffer2Shape = ctxt.lookup(operatorRepresentation[cls.dataIn2Name]).shape
        if not isBroadcastOperand(inputBuffer2Shape):
            return super().serializeTilingSolution(tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt,
                                                   operatorRepresentation)

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = [cls.dataIn1Name, cls.dataIn2Name, cls.dataOutName]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": [], "rowLen": []}
        replacementTypes = {"size": PointerClass(uint16_t), "rowLen": PointerClass(uint16_t)}

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            replacements["size"].append(np.prod(cube.dims))
            replacements["rowLen"].append(cube.dims[-1])

            # The vector's tile is the trailing slice of the output cube: same
            # extent and offset on the last axis, a single element on every other.
            rank = len(inputBuffer2Shape)
            broadcastOffset = tuple([0] * (rank - 1) + [cube.offset[-1]])
            broadcastDims = tuple([1] * (rank - 1) + [cube.dims[-1]])
            broadcastCube = HyperRectangle(broadcastOffset, broadcastDims)

            inputLoadSchedule.append({cls.dataIn1Name: cube, cls.dataIn2Name: broadcastCube})
            outputLoadSchedule.append({cls.dataOutName: cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
