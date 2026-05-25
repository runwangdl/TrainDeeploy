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

# Per-pattern L1 (TCDM) budget in bytes. Each TileConstraint pattern is solved
# independently, so this budget only has to cover this op's live tiles:
#   accum_tile + gradient_tile + lazy_reset (4 bytes).
# Set this to the *usable* TCDM size for the accumulator pattern. If this op
# shares L1 with other live buffers, keep a margin (e.g. multiply by 0.8).
L1_BUDGET_BYTES = 128000


class InPlaceAccumulatorV2TileConstraint(TileConstraint):
    """Tile constraint for InPlaceAccumulatorV2.

    Does NOT inherit from BOPTileConstraint — the accumulator's tiling is
    determined independently based on its own size, not forced by the
    gradient's upstream tiling pattern.

    accum_buffer and gradient have the same shape; lazy_reset_grad is a
    scalar (1 element) and is pinned full.

    InPlaceAccumulatorV2 is elementwise (acc += grad): every element is
    independent, so tiling only adds DMA overhead and the sole objective is
    to minimize the tile count (largest possible tile). The DMA engine
    supports 1D and 2D transfers, which is exploited by the greedy pinning
    policy in addPolicyConstraint.
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

        outShape = ctxt.lookup(outName).shape
        outDims = [outShape] if isinstance(outShape, int) else outShape

        # Tie gradient to accumulator (same shape, same tiling).
        # Only tie data_out if it has the same shape as accum — data_out may be
        # a scalar [1] indicator (not the actual accumulated result), in which
        # case tying would force accum dim == 1 and block full-pinning.
        tieOutput = (len(dims) == len(outDims) and all(a == b for a, b in zip(dims, outDims)))

        for dim in range(len(dims)):
            accumDimVar = tilerModel.getTensorDimVar(tensorName = accumName, dimIdx = dim)
            gradDimVar = tilerModel.getTensorDimVar(tensorName = gradName, dimIdx = dim)

            tilerModel.addConstraint(accumDimVar == gradDimVar)

            if tieOutput:
                outDimVar = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = dim)
                tilerModel.addConstraint(accumDimVar == outDimVar)

        # If data_out is scalar and not tied, pin it full (like lazy_reset_grad).
        if not tieOutput:
            for idx, d in enumerate(outDims):
                outDimVar = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = idx)
                tilerModel.addConstraint(outDimVar == d)



        # lazy_reset_grad is a scalar flag — pin full so it is never tiled.
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
        """Pin all accumulator dims full via PerformanceHint.

        InPlaceAccumulatorV2 is elementwise — fewer tiles = less DMA overhead.
        Inner dims get higher priority so they are pinned first and dropped
        last if memory pressure requires tiling.
        """
        accumName = parseDict[cls.dataIn1Name]
        accumBuf = ctxt.lookup(accumName)
        shape = accumBuf.shape

        # Normalize 1D int shape to a list
        if isinstance(shape, int):
            shape = [shape]

        for dimIdx in range(len(shape) - 1, -1, -1):
            dimVar = tilerModel.getTensorDimVar(accumName, dimIdx)
            priority = 1000 + (len(shape) - dimIdx)
            tilerModel.addConstraint(dimVar == shape[dimIdx], strategy = PerformanceHint(priority = priority))

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