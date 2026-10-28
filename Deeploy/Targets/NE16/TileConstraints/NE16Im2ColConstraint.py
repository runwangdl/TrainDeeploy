# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp6): tile constraint for `NE16Im2ColU8` -- the column-buffer builder.

Tiled along the OUTPUT ROWS (whole rows, all `ch_in*K` column channels), like a conv: an output row
tile [ho, ho+Hs) needs input rows [ho*sy - pad_top, (ho+Hs-1)*sy - pad_top + kh) clipped to the
image, and the clipped-away part is the tile's own padding. Columns stay whole, so left/right padding
is always the node's. The kernel receives the tile's dims and pads per call, exactly as pulp-nn's
conv does under RQConv2DTileConstraint.

Why tiling matters here even though the whole column (39 KB) would fit L1: an UNTILED node keeps its
tensors at the default memory level (L2), and the cores then read/write L2 directly -- the first
exp6 run measured 142 k cycles for the column build that way, ~10x what the same gather costs from
L1. Tiling is what brings the input into L1 and lets the column be written there. -- QW
"""

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class NE16Im2ColU8TileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inName, outName = parseDict['data_in'], parseDict['data_out']
        for name in (inName, outName):
            tilerModel.addTensorDimToModel(ctxt, name)
        inB = tilerModel.getTensorDimVar(inName, 0)
        inC = tilerModel.getTensorDimVar(inName, 1)
        inH = tilerModel.getTensorDimVar(inName, 2)
        inW = tilerModel.getTensorDimVar(inName, 3)
        outB = tilerModel.getTensorDimVar(outName, 0)
        outH = tilerModel.getTensorDimVar(outName, 1)
        outW = tilerModel.getTensorDimVar(outName, 2)
        outK = tilerModel.getTensorDimVar(outName, 3)

        kh, kw = parseDict['ky_n'], parseDict['kx_n']
        sy, sx = parseDict['stride_y'], parseDict['stride_x']
        pt, pb, pl, pr = parseDict['pad_top'], parseDict['pad_bottom'], parseDict['pad_left'], parseDict['pad_right']
        inShape = ctxt.lookup(inName).shape

        tilerModel.addConstraint(outB == inB)
        tilerModel.addConstraint(inC == inC.Max())  # all input channels
        tilerModel.addConstraint(outK == outK.Max())  # all column channels
        tilerModel.addConstraint(outW == outW.Max())  # whole rows
        tilerModel.addConstraint(inW == inW.Max())
        # padding counts only when the input tile spans the full extent (same trick as the conv constraints)
        effH = inH + ((pt + pb) * (inH == inShape[2]))
        effW = inW + ((pl + pr) * (inW == inShape[3]))
        tilerModel.addConstraint(outH == (effH - kh) // sy + 1)
        tilerModel.addConstraint(outW == (effW - kw) // sx + 1)
        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inH = tilerModel.getTensorDimVar(parseDict['data_in'], 2)
        tilerModel.addConstraint(inH >= parseDict['ky_n'])
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, ['data_in', 'data_out'])
        r = operatorRepresentation
        kh, kw, sy, sx = r['ky_n'], r['kx_n'], r['stride_y'], r['stride_x']
        pt, pl = r['pad_top'], r['pad_left']
        _, C, H, W = (int(s) for s in ctxt.lookup(r['data_in']).shape)

        keys = ["in_rows", "in_cols", "out_rows", "out_cols", "pad_top", "pad_bottom", "pad_left", "pad_right"]
        replacements: Dict[str, List[int]] = {k: [] for k in keys}
        replacementTypes = {k: PointerClass(uint16_t) for k in keys[:4]}
        replacementTypes.update({k: PointerClass(uint8_t) for k in keys[4:]})

        inCubes = []
        for cube in outputCubes:
            (_, HOff, WOff, _) = cube.offset
            (_, Hs, Ws, _) = cube.dims

            def _span(off, size, s, k, pad, full):
                lo = off * s - pad
                hi = (off + size - 1) * s - pad + k
                i0, i1 = max(lo, 0), min(hi, full)
                return i0, i1, i0 - lo, max(0, hi - full)  # start, end, pad before, pad after

            ih0, ih1, p_t, p_b = _span(HOff, Hs, sy, kh, pt, H)
            iw0, iw1, p_l, p_r = _span(WOff, Ws, sx, kw, pl, W)
            replacements['in_rows'].append(ih1 - ih0)
            replacements['in_cols'].append(iw1 - iw0)
            replacements['out_rows'].append(Hs)
            replacements['out_cols'].append(Ws)
            replacements['pad_top'].append(p_t)
            replacements['pad_bottom'].append(p_b)
            replacements['pad_left'].append(p_l)
            replacements['pad_right'].append(p_r)
            inCubes.append(HyperRectangle((0, 0, ih0, iw0), (1, C, ih1 - ih0, iw1 - iw0)))

        inputLoadSchedule = [{"data_in": c} for c in inCubes]
        outputLoadSchedule = [{"data_out": c} for c in outputCubes]
        return VariableReplacementScheme(replacements, replacementTypes), \
            TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)


class NE16UntiledTileConstraint(TileConstraint):
    """QW (exp6): pin every listed input and the output whole -- for the small per-step preparation nodes
    (NE16DualPerturbEncode, NE16DualBias), like NE16WeightEncodeTileConstraint but with the input keys
    given as a CLASS attribute (the tiler calls serializeTilingSolution on the class, so no instance
    state may be needed). Tensors are tens of bytes; the consuming NE16 conv pins its weight whole anyway."""

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


class NE16DualPerturbEncodeTileConstraint(NE16UntiledTileConstraint):
    INPUT_KEYS = ['weight', 'wmul']


class NE16DualBiasTileConstraint(NE16UntiledTileConstraint):
    INPUT_KEYS = ['weight', 'wmul', 'add', 'bmul', 'mul']
