# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp8): untiled tile constraints for small per-step nodes of the dual-pass graph."""
from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class PULPUntiledTileConstraint(TileConstraint):
    """QW (dual_pass_fusion exp8): pin every listed input and the output whole (a copy of exp6's
    NE16UntiledTileConstraint for PULPOpen nodes). Input keys are a CLASS attribute: the tiler calls
    serializeTilingSolution on the class."""

    INPUT_KEYS: List[str] = []

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        for key in cls.INPUT_KEYS + ['data_out']:
            name = parseDict[key]
            tilerModel.addTensorDimToModel(ctxt, name)
            for dimIdx in range(len(ctxt.lookup(name).shape)):
                var = tilerModel.getTensorDimVar(tensorName = name, dimIdx = dimIdx)
                tilerModel.addConstraint(var == var.Max())
        return tilerModel

    @classmethod
    def constructSymbolicNodeRep(cls, tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        # Untiled: every dimension is a constant, so the parser's representation IS the symbolic one.
        # (The base class builds a fresh dict without `nodeName`, which computeTransientBuffersSize needs.)
        return parseDict.copy()

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation,
                                                                  cls.INPUT_KEYS + ['data_out'])

        def _whole(key: str) -> HyperRectangle:
            shape = tuple(int(s) for s in ctxt.lookup(operatorRepresentation[key]).shape)
            return HyperRectangle((0,) * len(shape), shape)

        inputLoadSchedule = [{k: _whole(k) for k in cls.INPUT_KEYS} for _ in outputCubes]
        outputLoadSchedule = [{"data_out": cube} for cube in outputCubes]
        return VariableReplacementScheme({}, {}), \
            TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)


class PerturbRademacherFixedTileConstraint(PULPUntiledTileConstraint):
    INPUT_KEYS = ['data_in']


class DualHalfTileConstraint(PULPUntiledTileConstraint):
    INPUT_KEYS = ['data_in']
