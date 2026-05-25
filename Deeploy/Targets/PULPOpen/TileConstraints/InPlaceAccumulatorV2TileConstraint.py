# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class InPlaceAccumulatorV2TileConstraint(TileConstraint):
    """Tile constraint for InPlaceAccumulatorV2.

    Does NOT inherit from BOPTileConstraint — the accumulator's tiling is
    determined independently based on its own size, not forced by the
    gradient's upstream tiling pattern.

    accum_buffer and gradient have the same shape; lazy_reset_grad is a
    scalar (1 element) and is pinned full.
    """

    dataIn1Name = 'accum_buffer'
    dataIn2Name = 'gradient'
    dataOutName = 'data_out'

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        accumName = parseDict[cls.dataIn1Name]
        gradName = parseDict[cls.dataIn2Name]
        outName = parseDict[cls.dataOutName]

        for bufferName in [accumName, gradName, outName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        accumShape = ctxt.lookup(accumName).shape
        dims = [accumShape] if isinstance(accumShape, int) else accumShape

        # Tie gradient and output to accumulator (same shape, same tiling)
        for dim in range(len(dims)):
            accumDimVar = tilerModel.getTensorDimVar(tensorName = accumName, dimIdx = dim)
            gradDimVar = tilerModel.getTensorDimVar(tensorName = gradName, dimIdx = dim)
            outDimVar = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = dim)

            tilerModel.addConstraint(accumDimVar == gradDimVar)
            tilerModel.addConstraint(accumDimVar == outDimVar)

        # Force spatial dims (index >= 2) to full size so that minimizeRectangle
        # can collapse them and DMA tiles stay rank ≤ 2.
        if len(dims) > 2:
            for dimIdx in range(2, len(dims)):
                dimVar = tilerModel.getTensorDimVar(tensorName = accumName, dimIdx = dimIdx)
                tilerModel.addConstraint(dimVar == dims[dimIdx])

        # lazy_reset_grad is a scalar flag — pin full size so it is not tiled.
        lazyResetName = parseDict['lazy_reset_grad']
        tilerModel.addTensorDimToModel(ctxt, lazyResetName)
        lazyShape = ctxt.lookup(lazyResetName).shape
        lazyDims = [lazyShape] if isinstance(lazyShape, int) else lazyShape
        for idx, dim in enumerate(lazyDims):
            dimVar = tilerModel.getTensorDimVar(lazyResetName, idx)
            tilerModel.addConstraint(dimVar == dim)

        return tilerModel

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """Pin dims to minimize tile count for InPlaceAccumulatorV2.

        Only pin dim1 for 2D tensors. 1D tensors cannot be pinned because
        their gradient is shared with upstream patterns (e.g. ReduceSum)
        that need to tile along dim0.
        """
        accumName = parseDict[cls.dataIn1Name]
        shape = ctxt.lookup(accumName).shape

        if isinstance(shape, int):
            return tilerModel

        if len(shape) == 1:
            # 1D bias vectors (e.g. [128] = 512 bytes) — try to pin full.
            # Use PerformanceHint so this is skipped if L1 budget is too tight.
            dim0Var = tilerModel.getTensorDimVar(accumName, 0)
            tilerModel.addConstraint(dim0Var == shape[0], strategy = PerformanceHint(priority = 1))
        elif len(shape) == 2:
            # 2D weight grads (e.g. [128, 128]) — pin dim1, tile along dim0
            dim1Var = tilerModel.getTensorDimVar(accumName, 1)
            tilerModel.addConstraint(dim1Var == shape[1])

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = [cls.dataIn1Name, cls.dataIn2Name, cls.dataOutName, 'lazy_reset_grad']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)
        outputBaseOffsets[cls.dataOutName] = inputBaseOffsets[cls.dataIn1Name]

        replacements = {"size": []}
        replacementTypes = {"size": PointerClass(uint16_t)}

        lazyResetShape = ctxt.lookup(operatorRepresentation['lazy_reset_grad']).shape
        lazyResetDims = (lazyResetShape,) if isinstance(lazyResetShape, int) else tuple(lazyResetShape)
        lazyResetCube = HyperRectangle((0,) * len(lazyResetDims), lazyResetDims)

        inputLoadSchedule = [{
            cls.dataIn1Name: cube,
            cls.dataIn2Name: cube,
            'lazy_reset_grad': lazyResetCube,
        } for cube in outputCubes]
        outputLoadSchedule = [{cls.dataOutName: out} for out in outputCubes]

        for cube in outputCubes:
            replacements["size"].append(int(np.prod(cube.dims)))

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
