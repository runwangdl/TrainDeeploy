# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import NodeBinding
from Deeploy.Targets.GAP9.Bindings import GAP9ClusterTransformer as ClusterTransformer
from Deeploy.Targets.Generic.TypeCheckers import ConvChecker
from Deeploy.Targets.NE16.Templates.Conv1xKTemplate import NE161xKConv2D_Template  # -- QW
from Deeploy.Targets.NE16.Templates.Conv3x3ChunkTemplate import NE163x3ChunkConv2D_Template  # -- QW
from Deeploy.Targets.NE16.Templates.ConvTemplate import NE16DenseConv2D_Template, NE16DWConv2D_Template, \
    NE16PWConv2D_Template, NE16RqntDenseConv2D_Template, NE16RqntDWConv2D_Template, NE16RqntPWConv2D_Template
from Deeploy.Targets.PULPOpen.TypeCheckers import PULPConvChecker

NE16RQSPWConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NE16RqntPWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NE16PWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NE16PWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NE16RQSDWConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NE16RqntDWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NE16DWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NE16DWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NE16RQSDenseConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NE16RqntDenseConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NE16DenseConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NE16DenseConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]


# QW (exp16a / STEP 2b): 1xK dense conv -> K pointwise dispatches. TWO inputs (data, pre-encoded
# weight) and an INT32 output: streamin requires quantization_bits==32, so the RequantShift is a
# separate cluster node and NE16 never requantises here. -- QW
NE161xKConv2DBindings = [
    NodeBinding(ConvChecker([PointerClass(data_in_type), PointerClass(uint8_t)], [PointerClass(int32_t)]),
                NE161xKConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
]


# QW (exp16b): 1xK dense conv -> ceil(K/3) DENSE 3x3 dispatches. Same shape as the 1xK binding:
# two inputs (data, pre-encoded weight) and an INT32 output (streamin needs quantization_bits==32).
NE163x3ChunkConv2DBindings = [
    NodeBinding(ConvChecker([PointerClass(data_in_type), PointerClass(uint8_t)], [PointerClass(int32_t)]),
                NE163x3ChunkConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
]
