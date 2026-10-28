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

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * operatorRepresentation['n_levels'] // 2

        operatorRepresentation['output_min'] = -(operatorRepresentation['n_levels'] // 2)
        operatorRepresentation['output_max'] = (operatorRepresentation['n_levels'] // 2) - 1

        # QW: the `rounding` flag must MATCH what PULPConvRequantMergePass would have done, because
        #     a RequantShift that stays standalone and one that gets fused into a RequantizedConv
        #     have to compute the same thing.
        #
        #     `_merge_conv_rq_fun` bakes `+div/2` into the add **only when the add is a CONSTANT**;
        #     for a runtime add (a perturbed bias -- an RQSPerturbRademacher output, which is the
        #     quantized-ZO case) it bakes nothing, and the fused kernel truncates, matching the host
        #     reference. This template, however, passed `rounding = 1` unconditionally.
        #
        #     For a constant add the two agree by construction: merged computes
        #     `(acc*mul + add + div/2) >> s` with the div/2 baked in and the kernel truncating;
        #     standalone computes the same because the kernel adds div/2 itself. For a VARIABLE add
        #     they diverged by exactly div/2 -- a systematic +0.5 LSB on every output element.
        #
        #     That bug was invisible until NE16 started leaving RequantShifts un-merged: with all
        #     convolutions fused, no standalone RequantShift with a runtime add ever existed. It is
        #     what made SpeechNet QZO's loss wrong on NE16 while every single-layer fixture passed
        #     (those fixtures bake the perturbed bias into a *constant*, so they took the agreeing
        #     branch). -- QW
        addBuffer = ctxt.lookup(operatorRepresentation['add'])
        operatorRepresentation['rqs_rounding'] = int(hasattr(addBuffer, "values") and addBuffer.values is not None)

        return ctxt, operatorRepresentation, []


referenceTemplate = _RequantShiftTemplate("""
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D
%>

// RequantShift (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    % if channels_first:
    RequantShift_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_NCHW(${data_in}, ${size}, ${mul}, ${add}, ${data_out}, ${log2Dstring}, ${channel_width}, ${input_offset}, ${output_offset}, ${output_min}, ${output_max}, ${rqs_rounding});
    % else:
    RequantShift_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_NHWC(${data_in}, ${size}, ${mul}, ${add}, ${data_out}, ${log2Dstring}, ${channels}, ${input_offset}, ${output_offset}, ${output_min}, ${output_max}, ${rqs_rounding});
    %endif
END_SINGLE_CORE
""")
