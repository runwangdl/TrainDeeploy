# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp5): tile constraint for `RequantizedDualConv`.

Modelled on RQConv2DTileConstraint with three differences:
  * six inputs (data_in, weight, mul, add, wmul, bmul) -- the two perturb multipliers travel with the
    conv, since the kernel perturbs internally;
  * the output carries 2*ch_out channels (A+ then A-), so out_ch == 2 * weight_out_ch;
  * output channels are NOT tiled. Splitting them would put A+ and A- in different tiles and undo
    the fusion, and the kernel's seed reproduction assumes the whole weight is one chunk-set.
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

_INPUTS = ['data_in', 'weight', 'mul', 'add', 'wmul', 'bmul']


class DualConv2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        names = {k: parseDict[k] for k in _INPUTS}
        outName = parseDict['data_out']
        strides, padding = parseDict["strides"], parseDict["pads"]

        for bufferName in list(names.values()) + [outName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inB = tilerModel.getTensorDimVar(tensorName = names['data_in'], dimIdx = 0)
        inH = tilerModel.getTensorDimVar(tensorName = names['data_in'], dimIdx = 1)
        inW = tilerModel.getTensorDimVar(tensorName = names['data_in'], dimIdx = 2)
        wO = tilerModel.getTensorDimVar(tensorName = names['weight'], dimIdx = 0)
        wH = tilerModel.getTensorDimVar(tensorName = names['weight'], dimIdx = 1)
        wW = tilerModel.getTensorDimVar(tensorName = names['weight'], dimIdx = 2)
        outB = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 0)
        outH = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 1)
        outW = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 2)
        outC = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 3)

        tilerModel.addConstraint(outB == inB)
        tilerModel.addConstraint(outC == 2 * wO)  # A+ and A- stacked on the channel axis
        for k in ('mul', 'add', 'wmul', 'bmul'):
            tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = names[k], dimIdx = 0) == wO)

        inputBuffer = ctxt.lookup(names['data_in'])
        effH = inH + ((padding[0] + padding[2]) * (inH == inputBuffer.shape[1]))
        effW = inW + ((padding[1] + padding[3]) * (inW == inputBuffer.shape[2]))
        tilerModel.addConstraint((outH == (effH - (wH - 1) - 1) // strides[0] + 1))
        tilerModel.addConstraint((outW == (effW - (wW - 1) - 1) // strides[1] + 1))
        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        inH = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inW = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inC = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)
        wO = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 0)
        wH = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        wW = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)
        wC = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 3)
        outC = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 3)
        strides = parseDict["strides"]

        # whole channel extent in one tile: the fusion needs A+ and A- together, and the seed
        # reproduction needs the whole weight as one chunk-set.
        tilerModel.addConstraint(wO == weightBuffer.shape[0])
        tilerModel.addConstraint(outC == outputBuffer.shape[3])
        tilerModel.addConstraint(inC == parseDict['ch_im_in'])
        tilerModel.addConstraint(wC == parseDict['ch_im_in'])
        tilerModel.addConstraint(inH >= parseDict['dim_kernel_x'])
        tilerModel.addConstraint(inW >= parseDict['dim_kernel_y'])
        tilerModel.addConstraint(wH == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(wW == parseDict['dim_kernel_y'])
        tilerModel.addConstraint((inH % strides[0]) == 0)
        tilerModel.addConstraint((inW % strides[1]) == 0)
        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        rep = parseDict.copy()
        rep['dim_im_in_x'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)
        rep['dim_kernel_x'] = tilerModel.getTensorDimVar(weightBuffer.name, 1)
        rep['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 2)
        return rep

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        addrNames = _INPUTS + ['data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)
        varWeight = operatorRepresentation['weight']
        varIn = operatorRepresentation['data_in']
        varOut = operatorRepresentation['data_out']
        wN, wH, wW, wC = ctxt.lookup(varWeight).shape
        pads, strides = operatorRepresentation['pads'], operatorRepresentation['strides']

        keys = ["dim_im_in_x", "dim_im_in_y", "dim_im_out_x", "dim_im_out_y", "ch_im_out", "padding_y_top",
                "padding_y_bottom", "padding_x_left", "padding_x_right"]
        replacements: Dict[str, List[int]] = {k: [] for k in keys}
        replacementTypes = {k: PointerClass(uint16_t) for k in keys[:5]}
        replacementTypes.update({k: PointerClass(uint8_t) for k in keys[5:]})

        inCubes, wCubes, vecCubes = [], [], []
        for cube in outputCubes:
            (_, HOffset, WOffset, COffset) = cube.offset
            (_, HSize, WSize, CSize) = cube.dims
            assert COffset == 0 and CSize == 2 * wN, "RequantizedDualConv: output channels must not be tiled"
            InCube, (pl, pr, pt, pb) = Conv2DTileConstraint.computeInputCube(kernelShape = (wH, wW),
                                                                            pads = pads,
                                                                            strides = strides,
                                                                            inputCSize = wC,
                                                                            outputCube = cube,
                                                                            inputDims = ctxt.lookup(varIn).shape,
                                                                            outputDims = ctxt.lookup(varOut).shape)
            replacements['dim_im_in_x'].append(InCube.dims[1])
            replacements['dim_im_in_y'].append(InCube.dims[2])
            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)
            replacements['ch_im_out'].append(CSize)
            replacements['padding_y_top'].append(pt)
            replacements['padding_y_bottom'].append(pb)
            replacements['padding_x_left'].append(pl)
            replacements['padding_x_right'].append(pr)
            inCubes.append(InCube)
            wCubes.append(HyperRectangle((0, 0, 0, 0), (wN, wH, wW, wC)))
            vecCubes.append(HyperRectangle((0,), (wN,)))

        inputLoadSchedule = [{
            "data_in": a, "weight": w, "mul": v, "add": v, "wmul": v, "bmul": v
        } for a, w, v in zip(inCubes, wCubes, vecCubes)]
        outputLoadSchedule = [{"data_out": out} for out in outputCubes]

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        return VariableReplacementScheme(replacements, replacementTypes), tilingSchedule
