# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    _invertPermutation, _permuteHyperRectangle
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class TransposeTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        inputShape = ctxt.lookup(inputBufferName).shape
        outputShape = ctxt.lookup(outputBufferName).shape
        perm = parseDict["perm"]

        # Spatial-view interpretation of the perm: it operates on the last
        # len(perm) dims of data_in and the last len(perm) dims of data_out.
        # MatMulLayer.computeShapes can left-pad the rank of one side without
        # touching the other when the same gs.Variable is shared between a
        # broadening (MatMul) and a non-broadening (Gemm/Transpose) consumer,
        # so the constraint indexing must offset by the per-side leading-batch
        # depth rather than assume rank == len(perm) == rank_other.  When all
        # ranks already match, offsets are 0 and behavior is unchanged.
        inputOffset = len(inputShape) - len(perm)
        outputOffset = len(outputShape) - len(perm)
        assert inputOffset >= 0 and outputOffset >= 0, (
            f"Transpose perm {perm} is longer than tensor ranks "
            f"data_in={inputShape}, data_out={outputShape}")

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # Map output spatial dims to input spatial dims via perm.
        for idx, perm_idx in enumerate(perm):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = outputBufferName,
                                           dimIdx = outputOffset + idx) == tilerModel.getTensorDimVar(
                                               tensorName = inputBufferName, dimIdx = inputOffset + perm_idx))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacementTypes = {}
        replacements: Dict[str, List[int]] = {}

        # Match the spatial-view interpretation in addGeometricalConstraint:
        # only the last len(perm) dims of data_in are actually transposed,
        # so emit exactly len(perm) dimLen_<i> replacement variables.
        numDims = len(operatorRepresentation['perm'])

        for dim in range(numDims):
            replacementTypes[f"dimLen_{dim}"] = PointerClass(uint16_t)
            replacements[f"dimLen_{dim}"] = []

        invPerm = _invertPermutation(operatorRepresentation['perm'])
        inputCubes = []
        for outCube in outputCubes:
            # Workaround: MatMulLayer.computeShapes may inject leading batch dims into
            # the Transpose output's ctxt shape (e.g. [K,N] → [1,K,N]) when A is 3-D.
            # The actual Transpose only operates on the spatial (last len(invPerm)) dims;
            # strip the extra leading batch dims before permuting so that inCube matches
            # the real data_in (weight) shape which is not broadened.
            numExtra = len(outCube.dims) - len(invPerm)
            if numExtra > 0:
                spatialCube = HyperRectangle(outCube.offset[numExtra:], outCube.dims[numExtra:])
                inCube = _permuteHyperRectangle(spatialCube, invPerm)
            else:
                inCube = _permuteHyperRectangle(outCube, invPerm)
            inputCubes.append(inCube)
            for i, dim in enumerate(inCube.dims):
                replacements[f"dimLen_{i}"].append(dim)

        inputLoadSchedule = [{"data_in": cube} for cube in inputCubes]
        outputLoadSchedule = [{"data_out": cube} for cube in outputCubes]
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
