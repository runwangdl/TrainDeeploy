# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (exp16a / STEP 2b): tile constraint for the 1xK -> K-pointwise NE16 decomposition.

What makes a 1xK node different from a pointwise one is the **halo**. Output columns
`[w0, w0+Wt)` are produced from

    tap 0  reading input [w0+0,  w0+0+Wt)
    tap 1  reading input [w0+1,  w0+1+Wt)
     ...
    tap K-1 reading    [w0+K-1, w0+K-1+Wt)
    -----------------------------------------
    union:             [w0, w0+Wt+K-1)

so each output tile of `Wt` columns needs `Wt + K - 1` input columns, overlapping the next tile by
`K-1`. `NE16PWConv2DTileConstraint` asserts `inputWidthVar == outputWidthVar` (true for a genuine
1x1 conv), which a 1xK node violates: taps 1..K-1 would read past the end of their input tile --
silently, into whatever else sits in L1.

This is exactly the relation `NE16DenseConv2DTileConstraint` already expresses for 3x3
(`outW == (effW - (3-1) - 1)//stride + 1`); here the constant 3 becomes `K`, taken from the node's
`ne16_taps` attribute, and applied along whichever axis carries the taps.

**Streamin imposes nothing here.** Deeploy emits `for (TILING_I) { DMA in; <execution block>;
DMA out; }` and the execution block is the whole template, so all K dispatches run inside ONE
tiling iteration on the same L1 buffers -- the output tile cannot be evicted between taps.
(Verified in the generated Network.c.) An earlier revision of this file claimed otherwise and used
that as a reason to forbid tiling; that was wrong.

The input is expected PRE-PADDED (padding is invalid in NE16's 1x1 mode and its guard in gvsoc
fsm.cpp:53 is commented out), so no padding is modelled here. -- QW
"""

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t, uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.NE16.Templates.ConvTemplate import NE162DDenseConvTemplate, NE162DPWConvTemplate, \
    getInputAddrOffset, ioStridesFromDimensions
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme

_NE16_SUBTILE_OUTPUT_HW = 3  # NE16 retires a 3x3 output patch per pass
_NE16_TP_OUT = 32


def _tapAxis(parseDict: Dict) -> int:
    """Which NHWC spatial axis carries the K taps: 2 for 1xK (W), 1 for Kx1 (H)."""
    kh, kw = (int(v) for v in parseDict['kernel_shape'])
    return 2 if kh == 1 else 1


class NE161xKConv2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inName, wName, outName = parseDict['data_in'], parseDict['weight'], parseDict['data_out']
        for name in (inName, wName, outName):
            tilerModel.addTensorDimToModel(ctxt, name)

        # QW: the halo is variant-specific -- taps-1 for the all-pointwise decomposition
        #     (exp16a), 3*(chunks-1)+2 for the 3x3-chunk one (exp16b). The parser computes it.
        halo = int(parseDict['ne16_halo'])  # -- QW
        axis = _tapAxis(parseDict)

        inBatch = tilerModel.getTensorDimVar(tensorName = inName, dimIdx = 0)
        outBatch = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 0)
        tilerModel.addConstraint(outBatch == inBatch)

        # QW (exp16b): the DENSE 3x3 chunk variant has a VERTICAL receptive field too -- a 3x3
        #     kernel needs one extra input row above and below each output row. Tiling the
        #     non-tapped axis would therefore need a per-tile 1-row halo AND per-tile padding
        #     (padding only applies at the true boundary), which this constraint does not model:
        #     leaving it unmodelled silently produced 14,491/19,712 wrong outputs. Pin that axis
        #     instead -- SpeechNet's non-tapped extent is 14 rows, so not tiling it is cheap.
        #     The all-pointwise variant (exp16a) has no vertical extent and is unaffected.
        pinOther = 'ne16_chunks' in parseDict  # -- QW
        for dimIdx in (1, 2):
            inVar = tilerModel.getTensorDimVar(tensorName = inName, dimIdx = dimIdx)
            outVar = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = dimIdx)
            if dimIdx == axis:
                tilerModel.addConstraint(outVar == inVar - halo)
            else:
                tilerModel.addConstraint(outVar == inVar)
                if pinOther:  # -- QW
                    tilerModel.addConstraint(outVar == outVar.Max())  # -- QW

        # Full input channels: NE16 would otherwise produce partial sums over cin that our
        # streamin chain (already carrying the K taps) has no room to also accumulate.
        inCh = tilerModel.getTensorDimVar(tensorName = inName, dimIdx = 3)
        tilerModel.addConstraint(inCh == inCh.Max())

        # The encoded weight holds ALL taps and is addressed by the template as
        # `weights_addr + j*weight_tap_bytes`; it must be present whole.
        wBuf = ctxt.lookup(wName)
        for dimIdx in range(len(wBuf.shape)):
            var = tilerModel.getTensorDimVar(tensorName = wName, dimIdx = dimIdx)
            tilerModel.addConstraint(var == var.Max())
        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        outName = parseDict['data_out']
        outH = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 1)
        outW = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 2)
        outC = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 3)

        # Same alignment preferences as the pointwise path: NE16 retires a 3x3 output patch and
        # TP_OUT=32 channels per pass, so body tiles that are multiples of those waste less.
        # PerformanceHints, never hard constraints -- a dimension smaller than the granularity
        # simply takes its whole extent.
        for key, var, gran, prio in (("dim_im_out_x", outH, _NE16_SUBTILE_OUTPUT_HW, 3),
                                     ("dim_im_out_y", outW, _NE16_SUBTILE_OUTPUT_HW, 2),
                                     ("ch_im_out", outC, _NE16_TP_OUT, 1)):
            if parseDict[key] > gran:
                tilerModel.addTileSizeDivisibleConstraint(parseDict, key, var, gran,
                                                          strategy = PerformanceHint(priority = prio))
            else:
                tilerModel.addConstraint(var == var.Max(), strategy = PerformanceHint(priority = prio))
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation,
                                                                  ['data_in', 'weight', 'data_out'])

        halo = int(operatorRepresentation['ne16_halo'])  # -- QW
        isDense3x3 = 'ne16_chunks' in operatorRepresentation  # -- QW
        axis = _tapAxis(operatorRepresentation)
        wBuf: VariableBuffer = ctxt.lookup(operatorRepresentation['weight'])

        keys = ["dim_im_in_x_stride", "dim_im_in_y_stride", "dim_im_out_x_stride", "dim_im_out_y_stride",
                "input_addr_offset", "nKo", "nKi", "nHo", "nWo", "bKo", "bKi", "bHo", "bWo", "bHi", "bWi"]
        replacements: Dict[str, List[int]] = {k: [] for k in keys}
        replacementTypes = {
            k: PointerClass(uint32_t if ("stride" in k or k == "input_addr_offset") else uint16_t)
            for k in keys
        }

        inputLoadSchedule, outputLoadSchedule = [], []
        for cube in outputCubes:
            bOff, hOff, wOff, _ = cube.offset
            bSz, hSz, wSz, cSz = cube.dims

            # THE HALO: the tapped axis needs `halo` extra input elements.
            inDims = [bSz, hSz, wSz, operatorRepresentation['ch_im_in']]
            inDims[axis] += halo
            inCube = HyperRectangle((bOff, hOff, wOff, 0), tuple(inDims))
            inputLoadSchedule.append({
                "data_in": inCube,
                # the whole encoded weight: the template indexes taps inside it
                "weight": HyperRectangle((0,) * len(wBuf.shape), tuple(wBuf.shape)),
            })
            outputLoadSchedule.append({"data_out": cube})

            inHSz, inWSz, inCSz = inDims[1], inDims[2], inDims[3]
            # Strides come from the INPUT tile's own width, so each row step skips the halo too.
            xStrideIn, yStrideIn = ioStridesFromDimensions(inWSz, inCSz, operatorRepresentation["input_bits"])
            xStrideOut, yStrideOut = ioStridesFromDimensions(wSz, cSz, operatorRepresentation["output_bits"])
            replacements["dim_im_in_x_stride"].append(xStrideIn)
            replacements["dim_im_in_y_stride"].append(yStrideIn)
            replacements["dim_im_out_x_stride"].append(xStrideOut)
            replacements["dim_im_out_y_stride"].append(yStrideOut)
            # QW: the counter formula and the padding are VARIANT-SPECIFIC.
            #     - all-pointwise (exp16a): 1x1 jobs, input pre-padded, so no NE16 padding and
            #       NE162DPWConvTemplate.getCounters (bHi = border - pad).
            #     - 3x3 chunks (exp16b): each job is a real 3x3 conv with NATIVE H padding, so it
            #       needs NE162DDenseConvTemplate.getCounters (bHi = border + 2 - pad, the +2
            #       being the 3x3 receptive field) AND the actual padding values.
            #     Using the pointwise pair for the 3x3 variant gives a border subtile that reads
            #     the wrong input extent -- which is what produced the +-1 errors concentrated at
            #     subtile borders.
            if isDense3x3:  # -- QW
                padB = int(operatorRepresentation['padding_y_bottom'])
                padR = int(operatorRepresentation['padding_x_right'])
                replacements["input_addr_offset"].append(
                    getInputAddrOffset(inWSz, yStrideIn, int(operatorRepresentation['padding_y_top']),
                                       int(operatorRepresentation['padding_x_left'])))
                counters = NE162DDenseConvTemplate.getCounters(inCSz, hSz, wSz, cSz, padB, padR,
                                                               operatorRepresentation)
            else:
                replacements["input_addr_offset"].append(0)  # input is pre-padded; no NE16 padding
                counters = NE162DPWConvTemplate.getCounters(inCSz, hSz, wSz, cSz, 0, 0, operatorRepresentation)
            for name, value in zip(["nKo", "nKi", "nHo", "nWo", "bKo", "bKi", "bHo", "bWo", "bHi", "bWi"],
                                   counters):
                replacements[name].append(value)

        return VariableReplacementScheme(replacements, replacementTypes), \
            TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
