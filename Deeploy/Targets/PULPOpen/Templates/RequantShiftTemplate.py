# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _RequantShiftTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation["signedI"] = data_in._type.referencedType.typeMin < 0
        operatorRepresentation["signedO"] = data_out._type.referencedType.typeMin < 0

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * operatorRepresentation['n_levels'] // 2

        if operatorRepresentation["signed"]:
            operatorRepresentation['output_min'] = -(operatorRepresentation['n_levels'] // 2)
            operatorRepresentation['output_max'] = (operatorRepresentation['n_levels'] // 2) - 1
        else:
            operatorRepresentation['output_min'] = 0
            operatorRepresentation['output_max'] = operatorRepresentation['n_levels'] - 1

        # QW: `rounding` must MATCH what PULPConvRequantMergePass would have done, because a
        #     RequantShift that stays standalone and one fused into a RequantizedConv must compute
        #     the same thing.
        #
        #     `_merge_conv_rq_fun` bakes `+div/2` into the add **only when the add is a CONSTANT**;
        #     for a runtime add (a perturbed bias -- an RQSPerturbRademacher output, i.e. the
        #     quantized-ZO case) it bakes nothing and the fused kernel truncates, matching the host
        #     reference. This template passed `rounding = 1` unconditionally.
        #
        #     Constant add: the two agree by construction -- merged is
        #     `(acc*mul + add + div/2) >> s` with div/2 baked and the kernel truncating; standalone
        #     is the same because the kernel adds div/2 itself. VARIABLE add: they diverged by
        #     exactly div/2, a systematic +0.5 LSB on every output element.
        #
        #     Invisible until NE16 started leaving RequantShifts un-merged -- with every conv fused,
        #     no standalone RequantShift with a runtime add ever existed. This is what made
        #     SpeechNet QZO's loss wrong on NE16 while every single-layer fixture passed (those bake
        #     the perturbed bias into a *constant*, so they took the agreeing branch). -- QW
        addBuffer = ctxt.lookup(operatorRepresentation['add'])
        operatorRepresentation['rqs_rounding'] = int(hasattr(addBuffer, "values") and addBuffer.values is not None)

        return ctxt, operatorRepresentation, []


referenceTemplate = _RequantShiftTemplate("""
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D

inSignage = "s" if signedI else "u"
outSignage = "s" if signedO else "u"
%>

// RequantShift (Name: ${nodeName}, Op: ${nodeOp})
    % if channels_first:
    RequantShift_${inSignage}${data_in_type.referencedType.typeWidth}_${outSignage}${data_out_type.referencedType.typeWidth}_NCHW(${data_in}, ${size}, ${mul}, ${add}, ${data_out}, ${log2Dstring}, ${channel_width}, 0, 0 , ${output_min}, ${output_max}, ${rqs_rounding});
    % else:
    RequantShift_${inSignage}${data_in_type.referencedType.typeWidth}_${outSignage}${data_out_type.referencedType.typeWidth}_NHWC(${data_in}, ${size}, ${mul}, ${add}, ${data_out}, ${log2Dstring}, ${channels}, 0, 0, ${output_min}, ${output_max}, ${rqs_rounding});
    %endif
""")
