# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.Generic.TileConstraints.BOPTileConstraint import BOPTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class InPlaceAccumulatorV2TileConstraint(BOPTileConstraint):
    """Tile constraint for InPlaceAccumulatorV2.

    Tiles accum_buffer and gradient together (same shape); lazy_reset_grad
    is a scalar (1 element) and is not tiled.
    """

    dataIn1Name = 'accum_buffer'
    dataIn2Name = 'gradient'
    dataOutName = 'data_out'

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = super().addGeometricalConstraint(tilerModel, parseDict, ctxt)

        # Force spatial dims (index >= 2) to full size so that minimizeRectangle
        # can collapse them and DMA tiles stay rank ≤ 2 (L3Dma pi_cl_ram_copy_2d limit).
        accumName = parseDict[cls.dataIn1Name]
        shape = ctxt.lookup(accumName).shape
        if not isinstance(shape, int) and len(shape) > 2:
            for dimIdx in range(2, len(shape)):
                dimVar = tilerModel.getTensorDimVar(tensorName = accumName, dimIdx = dimIdx)
                tilerModel.addConstraint(dimVar == shape[dimIdx])

        # lazy_reset_grad is a scalar flag — pin full size so it is not tiled.
        lazyResetName = parseDict['lazy_reset_grad']
        tilerModel.addTensorDimToModel(ctxt, lazyResetName)
        shape = ctxt.lookup(lazyResetName).shape
        dims = [shape] if isinstance(shape, int) else shape
        for idx, dim in enumerate(dims):
            dimVar = tilerModel.getTensorDimVar(lazyResetName, idx)
            tilerModel.addConstraint(dimVar == dim)

        return tilerModel

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """Tile the elementwise accumulator as coarsely as its rank allows.

        InPlaceAccumulatorV2 is elementwise (acc += grad), each element
        independent, and every tile pays a full staging round trip. Left free,
        the solver produces tiny tiles where most of the time is DMA overhead:

        - rank 1 (bias and norm gradients): keep whole. CCT-2's eight [128]
          biases were split into single-element tiles, 81% of its accumulation
          time. The largest rank-1 accumulator across the five benchmark models
          is 640 elements, so accum + gradient always fit in L1.
        - rank 2: pin dim 1, so tiles are whole rows.
        """
        accumName = parseDict[cls.dataIn1Name]
        shape = ctxt.lookup(accumName).shape

        if not isinstance(shape, int) and len(shape) == 1:
            dim0Var = tilerModel.getTensorDimVar(accumName, 0)
            tilerModel.addConstraint(dim0Var == shape[0])

        if not isinstance(shape, int) and len(shape) == 2:
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
