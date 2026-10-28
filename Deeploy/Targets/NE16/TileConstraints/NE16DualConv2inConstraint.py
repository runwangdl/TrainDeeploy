# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp7, step 7): tile constraint for `NE16DualConv2in`.

Tiles over the output height only. The width stays whole (the kernel builds one column per output row and its two
column buffers are sized 2 x Wo x K at codegen time), the channels stay whole (A+ and A- together; the encoded weight
[W+; W-] is one tensor), and x- gets exactly x+'s tile. mul / add2 / the encoded weight are loaded whole in every tile.
"""
from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme

_INPUTS = ['data_in', 'data_in_m', 'weight', 'mul', 'add']


def _inputs(rep: Dict) -> List[str]:
    """QW exp8: the stacked-input form has no data_in_m."""
    return [k for k in _INPUTS if k != 'data_in_m' or 'data_in_m' in rep]


class NE16DualConv2inTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        names = {k: parseDict[k] for k in _inputs(parseDict)}
        outName = parseDict['data_out']
        strides, pads = parseDict['strides'], parseDict['pads']
        kh = parseDict['dim_kernel_x']
        for bufferName in list(names.values()) + [outName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inB = tilerModel.getTensorDimVar(tensorName = names['data_in'], dimIdx = 0)
        inH = tilerModel.getTensorDimVar(tensorName = names['data_in'], dimIdx = 1)
        outB = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 0)
        outH = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 1)
        tilerModel.addConstraint(outB == inB)
        for d in range(4 if 'data_in_m' in names else 0):  # x- is tiled exactly like x+
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = names['data_in_m'], dimIdx = d) ==
                tilerModel.getTensorDimVar(tensorName = names['data_in'], dimIdx = d))
        inputBuffer = ctxt.lookup(names['data_in'])
        effH = inH + ((pads[0] + pads[2]) * (inH == inputBuffer.shape[1]))
        tilerModel.addConstraint(outH == (effH - (kh - 1) - 1) // strides[0] + 1)
        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # everything but the height is whole
        for key in ('data_in', 'data_out'):
            buf = ctxt.lookup(parseDict[key])
            for d in (0, 2, 3):
                var = tilerModel.getTensorDimVar(tensorName = buf.name, dimIdx = d)
                tilerModel.addConstraint(var == buf.shape[d])
        for key in ('weight', 'mul', 'add'):
            buf = ctxt.lookup(parseDict[key])
            for d in range(len(buf.shape)):
                var = tilerModel.getTensorDimVar(tensorName = buf.name, dimIdx = d)
                tilerModel.addConstraint(var == buf.shape[d])
        inH = tilerModel.getTensorDimVar(tensorName = parseDict['data_in'], dimIdx = 1)
        tilerModel.addConstraint(inH >= parseDict['dim_kernel_x'])
        tilerModel.addConstraint((inH % parseDict['strides'][0]) == 0)
        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        rep = parseDict.copy()
        rep['dim_im_in_x'] = tilerModel.getTensorDimVar(parseDict['data_in'], 1)
        return rep

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        addrNames = _inputs(operatorRepresentation) + ['data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)
        varIn, varOut = operatorRepresentation['data_in'], operatorRepresentation['data_out']
        inDims = ctxt.lookup(varIn).shape
        kh, kw = operatorRepresentation['dim_kernel_x'], operatorRepresentation['dim_kernel_y']
        ci = ctxt.lookup(varIn).shape[3]  # stored channels (2*Ci for the stacked form)
        wShape = ctxt.lookup(operatorRepresentation['weight']).shape
        vShape = ctxt.lookup(operatorRepresentation['mul']).shape
        aShape = ctxt.lookup(operatorRepresentation['add']).shape

        keys = ["dim_im_in_x", "dim_im_out_x", "padding_y_top", "padding_y_bottom"]
        replacements: Dict[str, List[int]] = {k: [] for k in keys}
        replacementTypes = {"dim_im_in_x": PointerClass(uint16_t), "dim_im_out_x": PointerClass(uint16_t),
                            "padding_y_top": PointerClass(uint8_t), "padding_y_bottom": PointerClass(uint8_t)}

        inCubes = []
        for cube in outputCubes:
            (_, HOffset, WOffset, COffset) = cube.offset
            (_, HSize, WSize, CSize) = cube.dims
            assert WOffset == 0 and COffset == 0, "NE16DualConv2in: only the height is tiled"
            InCube, (pl, pr, pt, pb) = Conv2DTileConstraint.computeInputCube(
                kernelShape = (kh, kw), pads = operatorRepresentation['pads'],
                strides = operatorRepresentation['strides'], inputCSize = ci, outputCube = cube,
                inputDims = inDims, outputDims = ctxt.lookup(varOut).shape)
            replacements['dim_im_in_x'].append(InCube.dims[1])
            replacements['dim_im_out_x'].append(HSize)
            replacements['padding_y_top'].append(pt)
            replacements['padding_y_bottom'].append(pb)
            inCubes.append(InCube)

        wCube = HyperRectangle(tuple(0 for _ in wShape), tuple(wShape))
        mCube = HyperRectangle(tuple(0 for _ in vShape), tuple(vShape))
        aCube = HyperRectangle(tuple(0 for _ in aShape), tuple(aShape))
        inputLoadSchedule = [{
            "data_in": a, "weight": wCube, "mul": mCube, "add": aCube,
            **({"data_in_m": a} if 'data_in_m' in operatorRepresentation else {})
        } for a in inCubes]
        outputLoadSchedule = [{"data_out": out} for out in outputCubes]
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        return VariableReplacementScheme(replacements, replacementTypes), tilingSchedule
