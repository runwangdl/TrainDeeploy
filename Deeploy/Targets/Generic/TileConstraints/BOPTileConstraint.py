# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class BOPTileConstraint(TileConstraint):
    """Tile constraint class for binary operators, i.e. operators that use two input tensors of equal dimensions
    """

    dataIn1Name = 'data_in_1'  #: str: Name of the first input tensor as defined by the operator's parser
    dataIn2Name = 'data_in_2'  #: str: Name of the second input tensor as defined by the operator's parser
    dataOutName = 'data_out'  #: str: Name of the output tensor as defined by the operator's parser

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBuffer1Name = parseDict[cls.dataIn1Name]
        inputBuffer2Name = parseDict[cls.dataIn2Name]
        outputBufferName = parseDict[cls.dataOutName]

        for bufferName in [inputBuffer1Name, inputBuffer2Name, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # An operand that is a single element is not tiled with the output: tying its
        # extents to the output's is what forces it to be stored at the output's size.
        # Either operand can be that scalar (a lowered graph may present a bias first).
        scalarIn1 = int(np.prod(ctxt.lookup(inputBuffer1Name).shape)) == 1
        scalarIn2 = int(np.prod(ctxt.lookup(inputBuffer2Name).shape)) == 1
        fullName, otherName = (inputBuffer2Name, inputBuffer1Name) if (scalarIn1 and not scalarIn2) else \
            (inputBuffer1Name, inputBuffer2Name)
        otherScalar = scalarIn1 if otherName == inputBuffer1Name else scalarIn2

        for dim in range(len(ctxt.lookup(fullName).shape)):
            fullDimVar = tilerModel.getTensorDimVar(tensorName = fullName, dimIdx = dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)

            if not otherScalar:
                otherDimVar = tilerModel.getTensorDimVar(tensorName = otherName, dimIdx = dim)
                tilerModel.addConstraint(fullDimVar == otherDimVar)
            tilerModel.addConstraint(fullDimVar == outputDimVar)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = [cls.dataIn1Name, cls.dataIn2Name, cls.dataOutName]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": []}

        replacementTypes = {"size": PointerClass(uint16_t)}

        for cube in outputCubes:
            newSize = np.prod(cube.dims)
            replacements["size"].append(newSize)

        inputLoadSchedule = []
        outputLoadSchedule = []

        # The scalar is transferred once at its own extent, not once per output tile at the
        # output's extent. Asking the DMA for the output cube out of a one-element buffer is
        # the same read-past-the-end that the Gemm broadcast bias had.
        in1Shape = ctxt.lookup(operatorRepresentation[cls.dataIn1Name]).shape
        in2Shape = ctxt.lookup(operatorRepresentation[cls.dataIn2Name]).shape
        scalarIn1 = int(np.prod(in1Shape)) == 1 and int(np.prod(in2Shape)) != 1
        scalarIn2 = int(np.prod(in2Shape)) == 1

        def _scalarCube(shape):
            # At least rank 1: a rank-0 scalar yields an empty offset tuple, and
            # minimizeRectangle indexes offset[0] unconditionally.
            cubeShape = tuple(shape) if len(shape) > 0 else (1,)
            return HyperRectangle(tuple(0 for _ in cubeShape), cubeShape)

        for cube in outputCubes:
            inputLoadSchedule.append({
                cls.dataIn1Name: _scalarCube(in1Shape) if scalarIn1 else cube,
                cls.dataIn2Name: _scalarCube(in2Shape) if scalarIn2 else cube,
            })

        for out in outputCubes:
            outputLoadSchedule.append({cls.dataOutName: out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
