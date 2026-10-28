# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

# QW: the forward MaxPool call below passes spatial dims as (H,W)=(y,x) to match the
#     PULP_MaxPool2d_*_HWC kernel's convention. The base template passed (x,y), which
#     is wrong for non-square / asymmetric pools (as in SpeechNet's 14x700 input). -- QW
# QW-TODO: referenceGradTemplate below still passes (x,y) to PULP_MaxPoolGrad2d_*_HWC —
#     reconcile to (y,x) when MaxPoolGrad is reworked (Part-4 argmax-mask). -- QW
referenceTemplate = NodeTemplate("""
// 2D Float MaxPool Channel Parallel (Name: ${nodeName}, Op: ${nodeOp})

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_MaxPool2d_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in},
        ${dim_kernel_y}, ${dim_kernel_x},
        ${stride_y}, ${stride_x},
        ref_${data_out}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );
    ref_${data_out}_${data_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
}
""")

referenceGradTemplate = NodeTemplate("""
// 2D Float MaxPoolGrad Channel Parallel (Name: ${nodeName}, Op: ${nodeOp})
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${x_in_type.typeName} ref_${data_out}_${x_in} = ${x_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {

    PULP_MaxPoolGrad2d_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ref_${data_out}_${x_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${dim_im_out_x}, ${dim_im_out_y},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${data_out}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );

    ref_${data_out}_${data_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
    ref_${data_out}_${x_in} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
    ref_${data_out}_${data_out} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
}
""")

# QW: Part-4 argmax-mask templates. -- QW
# MaxPoolArgmax: same (H,W)=(y,x) mapping as the forward MaxPool; output is the uint8
# within-window offset mask (pooled shape). -- QW
argmaxTemplate = NodeTemplate("""
// 2D Float MaxPoolArgmax Channel Parallel (Name: ${nodeName}, Op: ${nodeOp})

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_MaxPoolArgmax2d_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in},
        ${dim_kernel_y}, ${dim_kernel_x},
        ${stride_y}, ${stride_x},
        ref_${data_out}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );
    ref_${data_out}_${data_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
}
""")

# QW: mask-consuming MaxPoolGrad. inputs[1] (x_in) is now the uint8 offset mask (pooled
# shape), so its pointer advances by the POOLED size, and we call PULP_MaxPoolGradMask2d.
# Same dim mapping as referenceGradTemplate (dim_im_in = dY pooled, dim_im_out = dX). -- QW
referenceGradMaskTemplate = NodeTemplate("""
// 2D Float MaxPoolGradMask Channel Parallel (Name: ${nodeName}, Op: ${nodeOp})
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${x_in_type.typeName} ref_${data_out}_${x_in} = ${x_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {

    PULP_MaxPoolGradMask2d_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ref_${data_out}_${x_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${dim_im_out_x}, ${dim_im_out_y},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${data_out}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );

    ref_${data_out}_${data_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
    ref_${data_out}_${x_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
}
""")
